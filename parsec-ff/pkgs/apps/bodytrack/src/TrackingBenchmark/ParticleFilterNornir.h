//------------------------------------------------------------------------
//      ____                        _      _
//     / ___|____ _   _ ____   ____| |__  | |
//    | |   / ___| | | |  _  \/ ___|  _  \| |
//    | |___| |  | |_| | | | | |___| | | ||_|
//     \____|_|  \_____|_| |_|\____|_| |_|(_) Media benchmarks
//                           
//	  2006, Intel Corporation, licensed under Apache 2.0 
//
//  file :	 ParticleFilterNornir.h
//  author : Scott Ettinger - scott.m.ettinger@intel.com
//
//  description : Nornir parallelized version of the particle filter
//                object derived from ParticleFilterOMP.h
//                and written by Daniele De Sensi (d.desensi.software@gmail.com)
//		
//  modified : 
//--------------------------------------------------------------------------

#ifndef PARTICLEFILTERNORNIR_H
#define PARTICLEFILTERNORNIR_H

#if defined(HAVE_CONFIG_H)
# include "config.h"
#endif

#include <nornir/nornir.h>
#include "ParticleFilter.h"

template<class T> 
class ParticleFilterNornir : public ParticleFilter<T> {

	using ParticleFilter<T>:: mModel;
	using ParticleFilter<T>:: mWeights;
	using ParticleFilter<T>:: mParticles;
	using ParticleFilter<T>:: mNewParticles;
	using ParticleFilter<T>:: mBestParticle;
	using ParticleFilter<T>:: mNParticles;
	using ParticleFilter<T>:: mMinParticles;
	using ParticleFilter<T>:: mBins;  
	using ParticleFilter<T>:: mRnd;
	typedef typename ParticleFilter<T>::fpType fpType;
	typedef typename ParticleFilter<T>::Vectorf Vectorf;
private:
	nornir::ParallelFor* parallelFor;

protected:
	std::vector<int> mIndex;																//list of particles to regenerate

	//calculate particle weights - threaded version 
	void CalcWeights(std::vector<Vectorf > &particles);										//calculate particle weights based on model likelihood

	//New particle generation - threaded version 
	void GenerateNewParticles(int k);
public:
    void setParallelFor(nornir::ParallelFor* pf){
        parallelFor = pf;
    }
};

//Calculate particle weights (mWeights) and find highest likelihood particle. 
//computes an optimal annealing factor and scales the likelihoods. 
template<class T>
void ParticleFilterNornir<T>::CalcWeights(std::vector<Vectorf > &particles)
{
	std::vector<unsigned char> valid(particles.size());
	mBestParticle = 0;
	fpType total = 0, best = 0, minWeight = 1e30f, annealingFactor = 1;
	mWeights.resize(particles.size());

	int np = (int)particles.size(), j = 0;
    parallelFor->parallel_for(0, np, 1, 0, [&](const long long j, const long long thid)				//Nornir parallelized loop to compute log-likelihoods 
	{
		bool vflag;
        mWeights[j] = mModel->LogLikelihood(particles[j], vflag, thid);						//compute log-likelihood weights for each particle
		valid[j] = vflag ? 1 : 0;
	});


	uint i = 0;
	while(i < particles.size())
	{	if(!valid[i])																		//if not valid(model prior), remove the particle from the list
		{	particles[i] = particles[particles.size() - 1];
			mWeights[i] = mWeights[particles.size() - 1];
			valid[i] = valid[valid.size() - 1];
			particles.pop_back(); mWeights.pop_back(); valid.pop_back();
		}
		else
			minWeight = std::min(mWeights[i++], minWeight);									//find minimum log-likelihood
	}
	if((int)particles.size() < mMinParticles) return;										//bail out if not enough valid particles
	mWeights -= minWeight;																	//shift weights to zero for numerical stability
	if(mModel->StdDevs().size() > 1) 
		annealingFactor = BetaAnnealingFactor(mWeights, 0.5f);								//calculate annealing factor if more than 1 step
	for(i = 0; i < mWeights.size(); i++)
	{	double wa = annealingFactor * mWeights[i];
		mWeights[i] = (float)exp(wa);														//exponentiate log-likelihoods scaled by annealing factor
		total += mWeights[i];																//save sum of all weights
		if(i == 0 || mWeights[i] > best)													//find highest likelihood particle
		{	best = mWeights[i];
			mBestParticle = i;
		}
	}
	mWeights *= fpType(1.0) / total;														//normalize weights
}


//generate new particles distributed with std deviation given by the model annealing parameter - threaded
template<class T> 
void ParticleFilterNornir<T>::GenerateNewParticles(int k)
{	int p = 0;
	mNewParticles.resize(mNParticles);
	mIndex.resize(mNParticles);
	for(int i = 0; i < (int)mBins.size(); i++)										
		for(uint j = 0; j < mBins[i]; j++)													//index particles to be regenerated
			mIndex[p++] = i;
    
    parallelFor->parallel_for(0, mNParticles, 1, 1, [&](const long long i, const long long thid)            //distribute new particles randomly according to model stdDevs						
	{	mNewParticles[i] = mParticles[mIndex[i]];											//add new particle for each entry in each bin distributed randomly about duplicated particle
		AddGaussianNoise(mNewParticles[i], mModel->StdDevs()[k], mRnd[i]);
	});
}


#endif

