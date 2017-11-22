#ifndef _ENCODER_H_
#define _ENCODER_H_ 1

void Encode(config_t * conf);

#ifdef ENABLE_FF
void EncodeFF(config_t * conf);
#endif //ENABLE_FF

#ifdef ENABLE_NORNIR_NATIVE
void EncodeNornir(config_t * conf);
#endif // ENABLE_NORNIR_NATIVE

#endif /* !_ENCODER_H_ */
