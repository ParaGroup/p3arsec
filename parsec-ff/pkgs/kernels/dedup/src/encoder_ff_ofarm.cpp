/*
 * Decoder for dedup files
 *
 * Copyright 2010 Princeton University.
 * All rights reserved.
 *
 * Originally written by Minlan Yu.
 * Largely rewritten by Christian Bienia.
 * FastFlow version by Daniele De Sensi (d.desensi.software@gmail.com)
 */

/*
 * The pipeline model for Encode is Fragment->FragmentRefine->Deduplicate->Compress->Reorder
 * Each stage has basically three steps:
 * 1. fetch a group of items from the queue
 * 2. process the items
 * 3. put them in the queue for the next stage
 */

#ifdef ENABLE_FF

#include <assert.h>
#include <strings.h>
#include <math.h>
#include <limits.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <sys/stat.h>

extern "C" {
#include "util.h"
#include "dedupdef.h"
#include "encoder.h"
#include "debug.h"
#include "hashtable.h"
#include "config.h"
#include "rabin.h"
#include "mbuffer.h"

#include "queue.h"
#include "binheap.h"
#include "tree.h"
}

#include <iostream>

#ifdef ENABLE_NORNIR
#include <nornir/instrumenter.hpp>
#include <stdlib.h>
#include <iostream>
std::string getParametersPath(){
    return std::string(getenv("PARSECDIR")) + std::string("/parameters.xml");
}
nornir::Instrumenter* instr;
#endif //ENABLE_NORNIR

#include <ff/farm.hpp>
#include <ff/pipeline.hpp>

#ifdef ENABLE_GZIP_COMPRESSION
#include <zlib.h>
#endif //ENABLE_GZIP_COMPRESSION

#ifdef ENABLE_BZIP2_COMPRESSION
#include <bzlib.h>
#endif //ENABLE_BZIP2_COMPRESSION

#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif //ENABLE_PARSEC_HOOKS

#define INITIAL_SEARCH_TREE_SIZE 4096


//The configuration block defined in main
extern config_t * conf;

//Hash table data structure & utility functions
struct hashtable *cache;

static unsigned int hash_from_key_fn( void *k ) {
  //NOTE: sha1 sum is integer-aligned
  return ((unsigned int *)k)[0];
}

static int keys_equal_fn ( void *key1, void *key2 ) {
  return (memcmp(key1, key2, SHA1_LEN) == 0);
}

//Arguments to pass to each thread
struct thread_args {
  //file descriptor, first pipeline stage only
  int fd;
  //input file buffer, first pipeline stage & preloading only
  struct {
    void *buffer;
    size_t size;
  } input_file;
};


#ifdef ENABLE_STATISTICS
//Keep track of block granularity with 2^CHUNK_GRANULARITY_POW resolution (for statistics)
#define CHUNK_GRANULARITY_POW (7)
//Number of blocks to distinguish, CHUNK_MAX_NUM * 2^CHUNK_GRANULARITY_POW is biggest block being recognized (for statistics)
#define CHUNK_MAX_NUM (8*32)
//Map a chunk size to a statistics array slot
#define CHUNK_SIZE_TO_SLOT(s) ( ((s)>>(CHUNK_GRANULARITY_POW)) >= (CHUNK_MAX_NUM) ? (CHUNK_MAX_NUM)-1 : ((s)>>(CHUNK_GRANULARITY_POW)) )
//Get the average size of a chunk from a statistics array slot
#define SLOT_TO_CHUNK_SIZE(s) ( (s)*(1<<(CHUNK_GRANULARITY_POW)) + (1<<((CHUNK_GRANULARITY_POW)-1)) )
//Deduplication statistics (only used if ENABLE_STATISTICS is defined)
typedef struct {
  /* Cumulative sizes */
  size_t total_input; //Total size of input in bytes
  size_t total_dedup; //Total size of input without duplicate blocks (after global compression) in bytes
  size_t total_compressed; //Total size of input stream after local compression in bytes
  size_t total_output; //Total size of output in bytes (with overhead) in bytes

  /* Size distribution & other properties */
  unsigned int nChunks[CHUNK_MAX_NUM]; //Coarse-granular size distribution of data chunks
  unsigned int nDuplicates; //Total number of duplicate blocks
} stats_t;

//Initialize a statistics record
static void init_stats(stats_t *s) {
  int i;

  assert(s!=NULL);
  s->total_input = 0;
  s->total_dedup = 0;
  s->total_compressed = 0;
  s->total_output = 0;

  for(i=0; i<CHUNK_MAX_NUM; i++) {
    s->nChunks[i] = 0;
  }
  s->nDuplicates = 0;
}

//Merge two statistics records: s1=s1+s2
static void merge_stats(stats_t *s1, stats_t *s2) {
  int i;

  assert(s1!=NULL);
  assert(s2!=NULL);
  s1->total_input += s2->total_input;
  s1->total_dedup += s2->total_dedup;
  s1->total_compressed += s2->total_compressed;
  s1->total_output += s2->total_output;

  for(i=0; i<CHUNK_MAX_NUM; i++) {
    s1->nChunks[i] += s2->nChunks[i];
  }
  s1->nDuplicates += s2->nDuplicates;
}

//Print statistics
static void print_stats(stats_t *s) {
  const unsigned int unit_str_size = 7; //elements in unit_str array
  const char *unit_str[] = {"Bytes", "KB", "MB", "GB", "TB", "PB", "EB"};
  unsigned int unit_idx = 0;
  size_t unit_div = 1;

  assert(s!=NULL);

  //determine most suitable unit to use
  for(unit_idx=0; unit_idx<unit_str_size; unit_idx++) {
    unsigned int unit_div_next = unit_div * 1024;

    if(s->total_input / unit_div_next <= 0) break;
    if(s->total_dedup / unit_div_next <= 0) break;
    if(s->total_compressed / unit_div_next <= 0) break;
    if(s->total_output / unit_div_next <= 0) break;

    unit_div = unit_div_next;
  }

  printf("Total input size:              %14.2f %s\n", (float)(s->total_input)/(float)(unit_div), unit_str[unit_idx]);
  printf("Total output size:             %14.2f %s\n", (float)(s->total_output)/(float)(unit_div), unit_str[unit_idx]);
  printf("Effective compression factor:  %14.2fx\n", (float)(s->total_input)/(float)(s->total_output));
  printf("\n");

  //Total number of chunks
  unsigned int i;
  unsigned int nTotalChunks=0;
  for(i=0; i<CHUNK_MAX_NUM; i++) nTotalChunks+= s->nChunks[i];

  //Average size of chunks
  float mean_size = 0.0;
  for(i=0; i<CHUNK_MAX_NUM; i++) mean_size += (float)(SLOT_TO_CHUNK_SIZE(i)) * (float)(s->nChunks[i]);
  mean_size = mean_size / (float)nTotalChunks;

  //Variance of chunk size
  float var_size = 0.0;
  for(i=0; i<CHUNK_MAX_NUM; i++) var_size += (mean_size - (float)(SLOT_TO_CHUNK_SIZE(i))) *
                                             (mean_size - (float)(SLOT_TO_CHUNK_SIZE(i))) *
                                             (float)(s->nChunks[i]);

  printf("Mean data chunk size:          %14.2f %s (stddev: %.2f %s)\n", mean_size / 1024.0, "KB", sqrtf(var_size) / 1024.0, "KB");
  printf("Amount of duplicate chunks:    %14.2f%%\n", 100.0*(float)(s->nDuplicates)/(float)(nTotalChunks));
  printf("Data size after deduplication: %14.2f %s (compression factor: %.2fx)\n", (float)(s->total_dedup)/(float)(unit_div), unit_str[unit_idx], (float)(s->total_input)/(float)(s->total_dedup));
  printf("Data size after compression:   %14.2f %s (compression factor: %.2fx)\n", (float)(s->total_compressed)/(float)(unit_div), unit_str[unit_idx], (float)(s->total_dedup)/(float)(s->total_compressed));
  printf("Output overhead:               %14.2f%%\n", 100.0*(float)(s->total_output-s->total_compressed)/(float)(s->total_output));
}

//variable with global statistics
stats_t stats;
#endif //ENABLE_STATISTICS


//Simple write utility function
static int write_file(int fd, u_char type, u_long len, u_char * content) {
  if (xwrite(fd, &type, sizeof(type)) < 0){
    perror("xwrite:");
    EXIT_TRACE("xwrite type fails\n");
    return -1;
  }
  if (xwrite(fd, &len, sizeof(len)) < 0){
    EXIT_TRACE("xwrite content fails\n");
  }
  if (xwrite(fd, content, len) < 0){
    EXIT_TRACE("xwrite content fails\n");
  }
  return 0;
}

/*
 * Helper function that creates and initializes the output file
 * Takes the file name to use as input and returns the file handle
 * The output file can be used to write chunks without any further steps
 */
static int create_output_file(char *outfile) {
  int fd;

  //Create output file
  fd = open(outfile, O_CREAT|O_TRUNC|O_WRONLY|O_TRUNC, S_IRGRP | S_IWUSR | S_IRUSR | S_IROTH);
  if (fd < 0) {
    EXIT_TRACE("Cannot open output file.");
  }

  //Write header
  if (write_header(fd, conf->compress_type)) {
    EXIT_TRACE("Cannot write output file header.\n");
  }

  return fd;
}



/*
 * Helper function that writes a chunk to an output file depending on
 * its state. The function will write the SHA1 sum if the chunk has
 * already been written before, or it will write the compressed data
 * of the chunk if it has not been written yet.
 *
 * This function will block if the compressed data is not available yet.
 * This function might update the state of the chunk if there are any changes.
 */
//NOTE: The parallel version checks the state of each chunk to make sure the
//      relevant data is available. If it is not then the function waits.
static void write_chunk_to_file(int fd, chunk_t *chunk) {
  assert(chunk!=NULL);

  //Find original chunk
  if(chunk->header.isDuplicate) chunk = chunk->compressed_data_ref;

  pthread_mutex_lock(&chunk->header.lock);
  while(chunk->header.state == CHUNK_STATE_UNCOMPRESSED) {
    pthread_cond_wait(&chunk->header.update, &chunk->header.lock);
  }

  //state is now guaranteed to be either COMPRESSED or FLUSHED
  if(chunk->header.state == CHUNK_STATE_COMPRESSED) {
    //Chunk data has not been written yet, do so now
    write_file(fd, TYPE_COMPRESS, chunk->compressed_data.n, (u_char*) chunk->compressed_data.ptr);
    mbuffer_free(&chunk->compressed_data);
    chunk->header.state = CHUNK_STATE_FLUSHED;
  } else {
    //Chunk data has been written to file before, just write SHA1
    write_file(fd, TYPE_FINGERPRINT, SHA1_LEN, (unsigned char *)(chunk->sha1));
  }
  pthread_mutex_unlock(&chunk->header.lock);
}

int rf_win;
int rf_win_dataprocess;

/*
 * Computational kernel of compression stage
 *
 * Actions performed:
 *  - Compress a data chunk
 */
static void sub_Compress(chunk_t *chunk) {
    size_t n;
    int r;

    assert(chunk!=NULL);
    //compress the item and add it to the database
    pthread_mutex_lock(&chunk->header.lock);
    assert(chunk->header.state == CHUNK_STATE_UNCOMPRESSED);
    switch (conf->compress_type) {
      case COMPRESS_NONE:
        //Simply duplicate the data
        n = chunk->uncompressed_data.n;
        r = mbuffer_create(&chunk->compressed_data, n);
        if(r != 0) {
          EXIT_TRACE("Creation of compression buffer failed.\n");
        }
        //copy the block
        memcpy(chunk->compressed_data.ptr, chunk->uncompressed_data.ptr, chunk->uncompressed_data.n);
        break;
#ifdef ENABLE_GZIP_COMPRESSION
      case COMPRESS_GZIP:
        //Gzip compression buffer must be at least 0.1% larger than source buffer plus 12 bytes
        n = chunk->uncompressed_data.n + (chunk->uncompressed_data.n >> 9) + 12;
        r = mbuffer_create(&chunk->compressed_data, n);
        if(r != 0) {
          EXIT_TRACE("Creation of compression buffer failed.\n");
        }
        //compress the block
        r = compress((u_char*) chunk->compressed_data.ptr, &n, (u_char*) chunk->uncompressed_data.ptr, chunk->uncompressed_data.n);
        if (r != Z_OK) {
          EXIT_TRACE("Compression failed\n");
        }
        //Shrink buffer to actual size
        if(n < chunk->compressed_data.n) {
          r = mbuffer_realloc(&chunk->compressed_data, n);
          assert(r == 0);
        }
        break;
#endif //ENABLE_GZIP_COMPRESSION
#ifdef ENABLE_BZIP2_COMPRESSION
      case COMPRESS_BZIP2:
        //Bzip compression buffer must be at least 1% larger than source buffer plus 600 bytes
        n = chunk->uncompressed_data.n + (chunk->uncompressed_data.n >> 6) + 600;
        r = mbuffer_create(&chunk->compressed_data, n);
        if(r != 0) {
          EXIT_TRACE("Creation of compression buffer failed.\n");
        }
        //compress the block
        unsigned int int_n = n;
        r = BZ2_bzBuffToBuffCompress(chunk->compressed_data.ptr, &int_n, chunk->uncompressed_data.ptr, chunk->uncompressed_data.n, 9, 0, 30);
        n = int_n;
        if (r != BZ_OK) {
          EXIT_TRACE("Compression failed\n");
        }
        //Shrink buffer to actual size
        if(n < chunk->compressed_data.n) {
          r = mbuffer_realloc(&chunk->compressed_data, n);
          assert(r == 0);
        }
        break;
#endif //ENABLE_BZIP2_COMPRESSION
      default:
        EXIT_TRACE("Compression type not implemented.\n");
        break;
    }
    mbuffer_free(&chunk->uncompressed_data);

    chunk->header.state = CHUNK_STATE_COMPRESSED;
    pthread_cond_broadcast(&chunk->header.update);
    pthread_mutex_unlock(&chunk->header.lock);
    return;
}

/*
 * Pipeline stage function of compression stage
 *
 * Actions performed:
 *  - Dequeue items from compression queue
 *  - Execute compression kernel for each item
 *  - Enqueue each item into send queue
 */
class Compress{
private:
    int r;
    ringbuffer_t *send_buf;
#ifdef ENABLE_STATISTICS
    stats_t *thread_stats;
#endif
public:
    Compress(){
#ifdef ENABLE_STATISTICS
        thread_stats = (stats_t*) malloc(sizeof(stats_t));
        if(thread_stats == NULL) EXIT_TRACE("Memory allocation failed.\n");
        init_stats(thread_stats);
#endif //ENABLE_STATISTICS

        send_buf = (ringbuffer_t*) malloc(sizeof(ringbuffer_t));
        r = 0;
        r += ringbuffer_init(send_buf, ITEM_PER_INSERT);
        assert(r==0);
    }

#ifdef ENABLE_STATISTICS
    stats_t* getStats() const{return thread_stats;}
#endif

    std::vector<ringbuffer_t*> compute(std::vector<ringbuffer_t*> recv_buf_v) {
        std::vector<ringbuffer_t*> dataOut;
        for(ringbuffer_t* recv_buf : recv_buf_v){
            bool isLast = ringbuffer_isLast(recv_buf);
            while(!ringbuffer_isEmpty(recv_buf)){
                //fetch one item
                chunk_t * chunk = (chunk_t *)ringbuffer_remove(recv_buf);
                assert(chunk!=NULL);
                if(chunk->header.isDuplicate){
		        r = ringbuffer_insert(send_buf, chunk);
		        assert(r==0);

		        //put the item in the next queue for the write thread
		        if (ringbuffer_isFull(send_buf)) {
                    dataOut.push_back(send_buf);
		            send_buf = (ringbuffer_t*) malloc(sizeof(ringbuffer_t));
		            ringbuffer_init(send_buf, ITEM_PER_INSERT);
		        }
                  continue;
                }
      
                sub_Compress(chunk);

#ifdef ENABLE_STATISTICS
                thread_stats->total_compressed += chunk->compressed_data.n;
#endif //ENABLE_STATISTICS

                r = ringbuffer_insert(send_buf, chunk);
                assert(r==0);

                //put the item in the next queue for the write thread
                if (ringbuffer_isFull(send_buf)) {
                    dataOut.push_back(send_buf);
                    send_buf = (ringbuffer_t*) malloc(sizeof(ringbuffer_t));
                    ringbuffer_init(send_buf, ITEM_PER_INSERT);
                }
            }

            ringbuffer_destroy(recv_buf);
            free(recv_buf);
            if(isLast){
                //empty buffers
                ringbuffer_setLast(send_buf);
                dataOut.push_back(send_buf);
            }
        }
        return dataOut;
    }
};

/*
 * Computational kernel of deduplication stage
 *
 * Actions performed:
 *  - Calculate SHA1 signature for each incoming data chunk
 *  - Perform database lookup to determine chunk redundancy status
 *  - On miss add chunk to database
 *  - Returns chunk redundancy status
 */
static int sub_Deduplicate(chunk_t *chunk) {
  int isDuplicate;
  chunk_t *entry;

  assert(chunk!=NULL);
  assert(chunk->uncompressed_data.ptr!=NULL);

  SHA1_Digest(chunk->uncompressed_data.ptr, chunk->uncompressed_data.n, (unsigned char *)(chunk->sha1));

  //Query database to determine whether we've seen the data chunk before
  pthread_mutex_t *ht_lock = hashtable_getlock(cache, (void *)(chunk->sha1));
  pthread_mutex_lock(ht_lock);
  entry = (chunk_t *)hashtable_search(cache, (void *)(chunk->sha1));
  isDuplicate = (entry != NULL);
  chunk->header.isDuplicate = isDuplicate;
  if (!isDuplicate) {
    // Cache miss: Create entry in hash table and forward data to compression stage
    pthread_mutex_init(&chunk->header.lock, NULL);
    pthread_cond_init(&chunk->header.update, NULL);
    //NOTE: chunk->compressed_data.buffer will be computed in compression stage
    if (hashtable_insert(cache, (void *)(chunk->sha1), (void *)chunk) == 0) {
      EXIT_TRACE("hashtable_insert failed");
    }
  } else {
    // Cache hit: Skipping compression stage
    chunk->compressed_data_ref = entry;
    mbuffer_free(&chunk->uncompressed_data);
  }
  pthread_mutex_unlock(ht_lock);

  return isDuplicate;
}

/*
 * Pipeline stage function of deduplication stage
 *
 * Actions performed:
 *  - Take input data from fragmentation stages
 *  - Execute deduplication kernel for each data chunk
 *  - Route resulting package either to compression stage or to reorder stage, depending on deduplication status
 */

class Deduplicate{
private:
    int r;
    ringbuffer_t *send_buf;
#ifdef ENABLE_STATISTICS
    stats_t *thread_stats;
#endif //ENABLE_STATISTICS
public:
    Deduplicate(){
#ifdef ENABLE_STATISTICS
        thread_stats = (stats_t*) malloc(sizeof(stats_t));
        if(thread_stats == NULL) {
            EXIT_TRACE("Memory allocation failed.\n");
        }
        init_stats(thread_stats);
#endif //ENABLE_STATISTICS
        send_buf = (ringbuffer_t*) malloc(sizeof(ringbuffer_t));

        r=0;

        r += ringbuffer_init(send_buf, ITEM_PER_INSERT);
        assert(r==0);
    }

#ifdef ENABLE_STATISTICS
    stats_t* getStats() const{return thread_stats;}
#endif

    std::vector<ringbuffer_t*> compute(std::vector<ringbuffer_t*> recv_buf_v) {
        std::vector<ringbuffer_t*> dataOut;
        for(ringbuffer_t* recv_buf : recv_buf_v){
            bool isLast = ringbuffer_isLast(recv_buf);
            while(!ringbuffer_isEmpty(recv_buf)){
                //get one chunk
                chunk_t *chunk = (chunk_t *)ringbuffer_remove(recv_buf);
                assert(chunk!=NULL);

                //Do the processing
                int isDuplicate = sub_Deduplicate(chunk);

#ifdef ENABLE_STATISTICS
                if(isDuplicate) {
                    thread_stats->nDuplicates++;
                } else {
                    thread_stats->total_dedup += chunk->uncompressed_data.n;
                }
#endif //ENABLE_STATISTICS

	      r = ringbuffer_insert(send_buf, chunk);
	      assert(r==0);
	      if (ringbuffer_isFull(send_buf)) {
            dataOut.push_back(send_buf);
	        send_buf = (ringbuffer_t*) malloc(sizeof(ringbuffer_t));
	        ringbuffer_init(send_buf, ITEM_PER_INSERT);
	      }
            }

            ringbuffer_destroy(recv_buf);
            free(recv_buf);
            if(isLast){
                //empty buffers
                // Is sufficient to set as last the buffer to compress. 
                // Compress will generate the buffer marked as last for reorder stage.
                ringbuffer_setLast(send_buf);
                dataOut.push_back(send_buf);
            }
        }
        return dataOut;
    }
};

/*
 * Pipeline stage function and computational kernel of refinement stage
 *
 * Actions performed:
 *  - Take coarse chunks from fragmentation stage
 *  - Partition data block into smaller chunks with Rabin rolling fingerprints
 *  - Send resulting data chunks to deduplication stage
 *
 * Notes:
 *  - Allocates mbuffers for fine-granular chunks
 */
class FragmentRefine{
private:
    ringbuffer_t *send_buf;
    int r;

    u32int * rabintab;
    u32int * rabinwintab;
#ifdef ENABLE_STATISTICS
    stats_t *thread_stats;
#endif
public:
    FragmentRefine(){
        rabintab = (u32int*) malloc(256*sizeof rabintab[0]);
        rabinwintab = (u32int*) malloc(256*sizeof rabintab[0]);
        if(rabintab == NULL || rabinwintab == NULL) {
            EXIT_TRACE("Memory allocation failed.\n");
        }

        send_buf = (ringbuffer_t*) malloc(sizeof(ringbuffer_t));
        r=0;
        r += ringbuffer_init(send_buf, CHUNK_ANCHOR_PER_INSERT);
        assert(r==0);

#ifdef ENABLE_STATISTICS
        thread_stats = (stats_t*) malloc(sizeof(stats_t));
        if(thread_stats == NULL) {
            EXIT_TRACE("Memory allocation failed.\n");
        }
        init_stats(thread_stats);
#endif //ENABLE_STATISTICS
    }

    ~FragmentRefine(){        
        free(rabintab);
        free(rabinwintab);
    }

#ifdef ENABLE_STATISTICS
    stats_t* getStats() const{return thread_stats;}
#endif

    std::vector<ringbuffer_t*> compute(ringbuffer_t* recv_buf) {
        std::vector<ringbuffer_t*> dataOut;
        bool isLast = ringbuffer_isLast(recv_buf);

        while(!ringbuffer_isEmpty(recv_buf)){
            //get one item
            chunk_t *chunk = (chunk_t *)ringbuffer_remove(recv_buf);
            assert(chunk!=NULL);

            rabininit(rf_win, rabintab, rabinwintab);

            int split;
            sequence_number_t chcount = 0;
            do {
                //Find next anchor with Rabin fingerprint
                int offset = rabinseg((uchar*) chunk->uncompressed_data.ptr, chunk->uncompressed_data.n, rf_win, rabintab, rabinwintab);
                //Can we split the buffer?
                if(offset < chunk->uncompressed_data.n) {
                    //Allocate a new chunk and create a new memory buffer
                    chunk_t *temp = (chunk_t *)malloc(sizeof(chunk_t));
                    if(temp==NULL) EXIT_TRACE("Memory allocation failed.\n");
                    temp->header.state = chunk->header.state;
                    temp->sequence.l1num = chunk->sequence.l1num;

                    //split it into two pieces
                    r = mbuffer_split(&chunk->uncompressed_data, &temp->uncompressed_data, offset);
                    if(r!=0) EXIT_TRACE("Unable to split memory buffer.\n");

                    //Set correct state and sequence numbers
                    chunk->sequence.l2num = chcount;
                    chunk->isLastL2Chunk = FALSE;
                    chcount++;

    #ifdef ENABLE_STATISTICS
                    //update statistics
                    thread_stats->nChunks[CHUNK_SIZE_TO_SLOT(chunk->uncompressed_data.n)]++;
    #endif //ENABLE_STATISTICS

                    //put it into send buffer
                    r = ringbuffer_insert(send_buf, chunk);
                    assert(r==0);
                    if (ringbuffer_isFull(send_buf)) {
                        dataOut.push_back(send_buf);
                        send_buf = (ringbuffer_t*) malloc(sizeof(ringbuffer_t));
                        ringbuffer_init(send_buf, CHUNK_ANCHOR_PER_INSERT);
                    }
                    //prepare for next iteration
                    chunk = temp;
                    split = 1;
                } else {
                    //End of buffer reached, don't split but simply enqueue it
                    //Set correct state and sequence numbers
                    chunk->sequence.l2num = chcount;
                    chunk->isLastL2Chunk = TRUE;
                    chcount++;

    #ifdef ENABLE_STATISTICS
                    //update statistics
                    thread_stats->nChunks[CHUNK_SIZE_TO_SLOT(chunk->uncompressed_data.n)]++;
    #endif //ENABLE_STATISTICS

                    //put it into send buffer
                    r = ringbuffer_insert(send_buf, chunk);
                    assert(r==0);
                    if (ringbuffer_isFull(send_buf)) {
                        dataOut.push_back(send_buf);
                        send_buf = (ringbuffer_t*) malloc(sizeof(ringbuffer_t));
                        ringbuffer_init(send_buf, CHUNK_ANCHOR_PER_INSERT);
                    }
                    //prepare for next iteration
                    chunk = NULL;
                    split = 0;
                }
            } while(split);
        }
        ringbuffer_destroy(recv_buf);
        free(recv_buf);
        //drain buffer
        if(isLast) {
            ringbuffer_setLast(send_buf);
            dataOut.push_back(send_buf);
        }
        return dataOut;
    }
};

/*
 * Pipeline stage function of fragmentation stage
 *
 * Actions performed:
 *  - Read data from file (or preloading buffer)
 *  - Perform coarse-grained chunking
 *  - Send coarse chunks to refinement stages for further processing
 *
 * Notes:
 * This pipeline stage is a bottleneck because it is inherently serial. We
 * therefore perform only coarse chunking and pass on the data block as fast
 * as possible so that there are no delays that might decrease scalability.
 * With very large numbers of threads this stage will not be able to keep up
 * which will eventually limit scalability. A solution to this is to increase
 * the size of coarse-grained chunks with a comparable increase in total
 * input size.
 */
class Fragment: public ff::ff_node{
    struct thread_args *args;
    size_t nw;
public:
    Fragment(struct thread_args* targs, size_t numWorkers):
            args(targs), nw(numWorkers){;}

    void* svc(void *){
        size_t preloading_buffer_seek = 0;
        int fd = args->fd;
        int i;

        ringbuffer_t* send_buf = (ringbuffer_t*) malloc(sizeof(ringbuffer_t));
        sequence_number_t anchorcount = 0;
        int r;

        chunk_t *temp = NULL;
        chunk_t *chunk = NULL;
        u32int * rabintab = (u32int*) malloc(256*sizeof rabintab[0]);
        u32int * rabinwintab = (u32int*) malloc(256*sizeof rabintab[0]);
        if(rabintab == NULL || rabinwintab == NULL) {
            EXIT_TRACE("Memory allocation failed.\n");
        }

        r = ringbuffer_init(send_buf, ANCHOR_DATA_PER_INSERT);
        assert(r==0);

        rf_win_dataprocess = 0;
        rabininit(rf_win_dataprocess, rabintab, rabinwintab);

        //Sanity check
        if(MAXBUF < 8 * ANCHOR_JUMP) {
            printf("WARNING: I/O buffer size is very small. Performance degraded.\n");
            fflush(NULL);
        }

        //read from input file / buffer
        while (1) {
            size_t bytes_left; //amount of data left over in last_mbuffer from previous iteration

            //Check how much data left over from previous iteration resp. create an initial chunk
            if(temp != NULL) {
                bytes_left = temp->uncompressed_data.n;
            } else {
                bytes_left = 0;
            }

            //Make sure that system supports new buffer size
            if(MAXBUF+bytes_left > SSIZE_MAX) {
                EXIT_TRACE("Input buffer size exceeds system maximum.\n");
            }
            //Allocate a new chunk and create a new memory buffer
            chunk = (chunk_t *)malloc(sizeof(chunk_t));
            if(chunk==NULL) EXIT_TRACE("Memory allocation failed.\n");
            r = mbuffer_create(&chunk->uncompressed_data, MAXBUF+bytes_left);
            if(r!=0) {
                EXIT_TRACE("Unable to initialize memory buffer.\n");
            }
            if(bytes_left > 0) {
                //FIXME: Short-circuit this if no more data available

                //"Extension" of existing buffer, copy sequence number and left over data to beginning of new buffer
                chunk->header.state = CHUNK_STATE_UNCOMPRESSED;
                chunk->sequence.l1num = temp->sequence.l1num;

                //NOTE: We cannot safely extend the current memory region because it has already been given to another thread
                memcpy(chunk->uncompressed_data.ptr, temp->uncompressed_data.ptr, temp->uncompressed_data.n);
                mbuffer_free(&temp->uncompressed_data);
                free(temp);
                temp = NULL;
            } else {
                //brand new mbuffer, increment sequence number
                chunk->header.state = CHUNK_STATE_UNCOMPRESSED;
                chunk->sequence.l1num = anchorcount;
                anchorcount++;
            }
            //Read data until buffer full
            size_t bytes_read=0;
            if(conf->preloading) {
                size_t max_read = MIN(MAXBUF, args->input_file.size-preloading_buffer_seek);
                memcpy(chunk->uncompressed_data.ptr+bytes_left, args->input_file.buffer+preloading_buffer_seek, max_read);
                bytes_read = max_read;
                preloading_buffer_seek += max_read;
            } else {
                while(bytes_read < MAXBUF) {
                    r = read(fd, chunk->uncompressed_data.ptr+bytes_left+bytes_read, MAXBUF-bytes_read);
                    if(r<0) switch(errno) {
                    case EAGAIN:
                        EXIT_TRACE("I/O error: No data available\n");break;
                    case EBADF:
                        EXIT_TRACE("I/O error: Invalid file descriptor\n");break;
                    case EFAULT:
                        EXIT_TRACE("I/O error: Buffer out of range\n");break;
                    case EINTR:
                        EXIT_TRACE("I/O error: Interruption\n");break;
                    case EINVAL:
                        EXIT_TRACE("I/O error: Unable to read from file descriptor\n");break;
                    case EIO:
                        EXIT_TRACE("I/O error: Generic I/O error\n");break;
                    case EISDIR:
                        EXIT_TRACE("I/O error: Cannot read from a directory\n");break;
                    default:
                        EXIT_TRACE("I/O error: Unrecognized error\n");break;
                    }
                    if(r==0) break;
                    bytes_read += r;
                }
            }
            //No data left over from last iteration and also nothing new read in, simply clean up and quit
            if(bytes_left + bytes_read == 0) {
                mbuffer_free(&chunk->uncompressed_data);
                free(chunk);
                chunk = NULL;
                break;
            }
            //Shrink buffer to actual size
            if(bytes_left+bytes_read < chunk->uncompressed_data.n) {
                r = mbuffer_realloc(&chunk->uncompressed_data, bytes_left+bytes_read);
                assert(r == 0);
            }
            //Check whether any new data was read in, enqueue last chunk if not
            if(bytes_read == 0) {
                //put it into send buffer
                r = ringbuffer_insert(send_buf, chunk);
                assert(r==0);
                //NOTE: No need to empty a full send_buf, we will break now and pass everything on to the queue
                break;
            }
            //partition input block into large, coarse-granular chunks
            int split;
            do {
                split = 0;
                //Try to split the buffer at least ANCHOR_JUMP bytes away from its beginning
                if(ANCHOR_JUMP < chunk->uncompressed_data.n) {
                    int offset = rabinseg(chunk->uncompressed_data.ptr + ANCHOR_JUMP, chunk->uncompressed_data.n - ANCHOR_JUMP, rf_win_dataprocess, rabintab, rabinwintab);
                    //Did we find a split location?
                    if(offset == 0) {
                        //Split found at the very beginning of the buffer (should never happen due to technical limitations)
                        assert(0);
                        split = 0;
                    } else if(offset + ANCHOR_JUMP < chunk->uncompressed_data.n) {
                        //Split found somewhere in the middle of the buffer
                        //Allocate a new chunk and create a new memory buffer
                        temp = (chunk_t *)malloc(sizeof(chunk_t));
                        if(temp==NULL) EXIT_TRACE("Memory allocation failed.\n");

                        //split it into two pieces
                        r = mbuffer_split(&chunk->uncompressed_data, &temp->uncompressed_data, offset + ANCHOR_JUMP);
                        if(r!=0) EXIT_TRACE("Unable to split memory buffer.\n");
                        temp->header.state = CHUNK_STATE_UNCOMPRESSED;
                        temp->sequence.l1num = anchorcount;
                        anchorcount++;

                        //put it into send buffer
                        r = ringbuffer_insert(send_buf, chunk);
                        assert(r==0);

                        //send a group of items into the next queue in round-robin fashion
                        if(ringbuffer_isFull(send_buf)) {
                            ff_send_out((void*) send_buf);
                            send_buf = (ringbuffer_t*) malloc(sizeof(ringbuffer_t));
                            r = ringbuffer_init(send_buf, ANCHOR_DATA_PER_INSERT);
                        }
                        //prepare for next iteration
                        chunk = temp;
                        temp = NULL;
                        split = 1;
                    } else {
                        //Due to technical limitations we can't distinguish the cases "no split" and "split at end of buffer"
                        //This will result in some unnecessary (and unlikely) work but yields the correct result eventually.
                        temp = chunk;
                        chunk = NULL;
                        split = 0;
                    }
                } else {
                    //NOTE: We don't process the stub, instead we try to read in more data so we might be able to find a proper split.
                    //      Only once the end of the file is reached do we get a genuine stub which will be enqueued right after the read operation.
                    temp = chunk;
                    chunk = NULL;
                    split = 0;
                }
            } while(split);
        }
        //drain buffer
        ringbuffer_setLast(send_buf);
        while(!ff_send_out((void*) send_buf)){;}
        // Send the last buffer also to the other workers (nw - 1)
        for(size_t i = 1; i < nw; i++){
            send_buf = (ringbuffer_t*) malloc(sizeof(ringbuffer_t));
            r = ringbuffer_init(send_buf, ANCHOR_DATA_PER_INSERT);
            ringbuffer_setLast(send_buf);
            while(!ff_send_out((void*) send_buf)){;}
        }

        free(rabintab);
        free(rabinwintab);
        return EOS;
    }
};


/*
 * Pipeline stage function of reorder stage
 *
 * Actions performed:
 *  - Receive chunks from compression and deduplication stage
 *  - Check sequence number of each chunk to determine correct order
 *  - Cache chunks that arrive out-of-order until predecessors are available
 *  - Write chunks in-order to file (or preloading buffer)
 *
 * Notes:
 *  - This function blocks if the compression stage has not finished supplying
 *    the compressed data for a duplicate chunk.
 */
class Reorder: public ff::ff_node{
private:
    int fd;
public:
    Reorder():fd(create_output_file(conf->outfile)){;}

    ~Reorder(){
        close(fd);
    }

    void *svc(void * task) {
        ringbuffer_t* recv_buf = (ringbuffer_t*) task;
        while(!ringbuffer_isEmpty(recv_buf)) {
#ifdef ENABLE_NORNIR
          instr->begin();
#endif //ENABLE_NORNIR
            chunk_t * chunk = (chunk_t*)ringbuffer_remove(recv_buf);
            if (chunk == NULL){break;}
            write_chunk_to_file(fd, chunk);
#ifdef ENABLE_NORNIR
        instr->end();
#endif //ENABLE_NORNIR
        }
        ringbuffer_destroy(recv_buf);
        free(recv_buf);
        return GO_ON;
    }
};

#if GRAIN != 1
#error "To use ofarm version, GRAIN must be set to 1."
#endif

class CollapsedPipeline: public ff::ff_node{
private:
    FragmentRefine fr;
    Deduplicate d;
    Compress c;
public:

#ifdef ENABLE_STATISTICS
    stats_t* getFragmentRefineStats() const{return fr.getStats();}
    stats_t* getDeduplicateStats() const{return d.getStats();}
    stats_t* getCompressStats() const{return c.getStats();}
#endif

    void* svc(void* task){
        std::vector<ringbuffer_t*> toReorder = c.compute(d.compute(fr.compute((ringbuffer_t*) task)));
		size_t numChunks = toReorder.size() * CHUNK_ANCHOR_PER_INSERT;
		ringbuffer_t* send_buf = (ringbuffer_t*) malloc(sizeof(ringbuffer_t));
		ringbuffer_init(send_buf, numChunks);
		chunk_t* chunk = NULL;
        for(size_t i = 0; i < toReorder.size(); i++){
			while(!ringbuffer_isEmpty(toReorder[i]) &&
                 (chunk = (chunk_t*)ringbuffer_remove(toReorder[i]))) {
			    ringbuffer_insert(send_buf, chunk);
			}
			free(toReorder[i]);
        }
        return send_buf;
    }
};

static void runSingleFarm(config_t * conf, struct thread_args* data_process_args,
        stats_t **threads_anchor_rv, stats_t **threads_chunk_rv, stats_t **threads_compress_rv){
    ff::ff_ofarm ofarm;
    std::vector<ff::ff_node*> workers;
    for(size_t i = 0; i < conf->nthreads; i++){workers.push_back(new CollapsedPipeline());}
    ofarm.setEmitterF(new Fragment(data_process_args, conf->nthreads));
    ofarm.add_workers(workers);
    ofarm.setCollectorF(new Reorder());
    //ofarm.cleanup_all();
    ofarm.run_and_wait_end();
#ifdef ENABLE_STATISTICS
    for(size_t i = 0; i < conf->nthreads; i++){
        CollapsedPipeline* p = (CollapsedPipeline*) workers.at(i);
        threads_anchor_rv[i] = p->getFragmentRefineStats();
        threads_chunk_rv[i] = p->getDeduplicateStats();
        threads_compress_rv[i] = p->getCompressStats();
    }
#endif
}

/*--------------------------------------------------------------------------*/
/* Encode
 * Compress an input stream
 *
 * Arguments:
 *   conf:    Configuration parameters
 *
 */
void EncodeFF(config_t * _conf) {
  struct stat filestat;
  int32 fd;

  conf = _conf;

#ifdef ENABLE_STATISTICS
  init_stats(&stats);
#endif

#ifdef ENABLE_NORNIR
  instr = new nornir::Instrumenter(getParametersPath(), 1, NULL, true);
#endif //ENABLE_NORNIR

  //Create chunk cache
  cache = hashtable_create(65536, hash_from_key_fn, keys_equal_fn, FALSE);
  if(cache == NULL) {
    printf("ERROR: Out of memory\n");
    exit(1);
  }

  struct thread_args data_process_args;
  int i;

  assert(!mbuffer_system_init());

  /* src file stat */
  if (stat(conf->infile, &filestat) < 0) 
      EXIT_TRACE("stat() %s failed: %s\n", conf->infile, strerror(errno));

  if (!S_ISREG(filestat.st_mode)) 
    EXIT_TRACE("not a normal file: %s\n", conf->infile);
#ifdef ENABLE_STATISTICS
  stats.total_input = filestat.st_size;
#endif //ENABLE_STATISTICS

  /* src file open */
  if((fd = open(conf->infile, O_RDONLY | O_LARGEFILE)) < 0) 
    EXIT_TRACE("%s file open error %s\n", conf->infile, strerror(errno));

  //Load entire file into memory if requested by user
  void *preloading_buffer = NULL;
  if(conf->preloading) {
    size_t bytes_read=0;
    int r;

    preloading_buffer = malloc(filestat.st_size);
    if(preloading_buffer == NULL)
      EXIT_TRACE("Error allocating memory for input buffer.\n");

    //Read data until buffer full
    while(bytes_read < filestat.st_size) {
      r = read(fd, preloading_buffer+bytes_read, filestat.st_size-bytes_read);
      if(r<0) switch(errno) {
        case EAGAIN:
          EXIT_TRACE("I/O error: No data available\n");break;
        case EBADF:
          EXIT_TRACE("I/O error: Invalid file descriptor\n");break;
        case EFAULT:
          EXIT_TRACE("I/O error: Buffer out of range\n");break;
        case EINTR:
          EXIT_TRACE("I/O error: Interruption\n");break;
        case EINVAL:
          EXIT_TRACE("I/O error: Unable to read from file descriptor\n");break;
        case EIO:
          EXIT_TRACE("I/O error: Generic I/O error\n");break;
        case EISDIR:
          EXIT_TRACE("I/O error: Cannot read from a directory\n");break;
        default:
          EXIT_TRACE("I/O error: Unrecognized error\n");break;
      }
      if(r==0) break;
      bytes_read += r;
    }
    data_process_args.input_file.size = filestat.st_size;
    data_process_args.input_file.buffer = preloading_buffer;
  }

  data_process_args.fd = fd;

  // Statistics
  stats_t *threads_anchor_rv[conf->nthreads];
  stats_t *threads_chunk_rv[conf->nthreads];
  stats_t *threads_compress_rv[conf->nthreads];

#ifdef ENABLE_PARSEC_HOOKS
    __parsec_roi_begin();
#endif
    runSingleFarm(conf, &data_process_args, threads_anchor_rv, threads_chunk_rv, threads_compress_rv);
#ifdef ENABLE_PARSEC_HOOKS
  __parsec_roi_end();
#endif
#ifdef ENABLE_NORNIR
    instr->terminate();
    printf("riff.time|%d\n", instr->getExecutionTime());
    printf("riff.iterations|%d\n", instr->getTotalTasks());
    delete instr;
#endif //ENABLE_NORNIR

#ifdef ENABLE_STATISTICS
  //Merge everything into global `stats' structure
  for(i=0; i<conf->nthreads; i++) {
    merge_stats(&stats, threads_anchor_rv[i]);
    free(threads_anchor_rv[i]);
  }
  for(i=0; i<conf->nthreads; i++) {
    merge_stats(&stats, threads_chunk_rv[i]);
    free(threads_chunk_rv[i]);
  }
  for(i=0; i<conf->nthreads; i++) {
    merge_stats(&stats, threads_compress_rv[i]);
    free(threads_compress_rv[i]);
  }
#endif //ENABLE_STATISTICS

  //clean up after preloading
  if(conf->preloading) {
    free(preloading_buffer);
  }

  /* clean up with the src file */
  if (conf->infile != NULL)
    close(fd);

  assert(!mbuffer_system_destroy());

  hashtable_destroy(cache, TRUE);

#ifdef ENABLE_STATISTICS
  /* dest file stat */
  if (stat(conf->outfile, &filestat) < 0) 
      EXIT_TRACE("stat() %s failed: %s\n", conf->outfile, strerror(errno));
  stats.total_output = filestat.st_size;

  //Analyze and print statistics
  if(conf->verbose) print_stats(&stats);
#endif //ENABLE_STATISTICS
}

#endif //ENABLE_FF
