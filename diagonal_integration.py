# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 10:30:48 2022

@author: Yang Xu

diagonal integration
"""

import numpy as np
import pandas as pd

import scot as SCOT
from pamona import Pamona
from unioncom import UnionCom

import umap
import scanpy as sc

##Note
##Please use cross_domain_simulation.py to generate data domain1, domain2, y1, and y2
##Please follow github of each diagonal method for installment
##We modified code of cross-modal-AE to serve specific need in this analysis 

##-----------------------------------------------------------------------------
##diagonal integration via SCOT
scot_aligner=SCOT.SCOT(domain1, domain2)
k= 50
e= 1e-3
normalize=False
aligned_domain1, aligned_domain2, _ = scot_aligner.align(k=k, e=e, normalize=normalize)

##visualization
y_pred = np.concatenate((aligned_domain1, aligned_domain2),axis=0)
umaps = umap.UMAP(n_neighbors=15, min_dist=0.2, n_components=2,
                  metric="correlation").fit(y_pred)
embedding = umaps.transform(y_pred)
embedding = pd.DataFrame(embedding)
adata = sc.AnnData(X=y_pred.copy())

adata.obsm['X_umap']=embedding.iloc[:,:2].values
adata.obs['cell_type']=y1.tolist()+y2.tolist()
sc.pl.umap(adata, color='cell_type', title='',size=75)
adata.obs['Source']=np.repeat(['Domain1','Domain2'],repeats=[len(y1),len(y2)])
sc.pl.umap(adata, color='Source', title='',size=75)

##-----------------------------------------------------------------------------
##diagonal integration via UnionCom
uc = UnionCom.UnionCom()
y_pred = uc.fit_transform(dataset=[domain1, domain2])
y_pred = np.concatenate((y_pred[0], y_pred[1]),axis=0)

##visualization
umaps = umap.UMAP(n_neighbors=15, min_dist=0.2, n_components=2,
                  metric="correlation").fit(y_pred)
embedding = umaps.transform(y_pred)
embedding = pd.DataFrame(embedding)
adata = sc.AnnData(X=y_pred.copy())

adata.obsm['X_umap']=embedding.iloc[:,:2].values
adata.obs['cell_type']=y1.tolist()+y2.tolist()
sc.pl.umap(adata, color='cell_type', title='',size=75)
adata.obs['Source']=np.repeat(['Domain1','Domain2'],repeats=[len(y1),len(y2)])
sc.pl.umap(adata, color='Source', title='',size=75)

##-----------------------------------------------------------------------------
##diagonal integration via Pamona
Pa = Pamona.Pamona(Lambda=10, output_dim=5)
integrated_data, T = Pa.run_Pamona([domain1, domain2])
y_pred = np.concatenate((integrated_data[0], integrated_data[1]),axis=0)

##visualization
umaps = umap.UMAP(n_neighbors=15, min_dist=0.2, n_components=2,
                  metric="correlation").fit(y_pred)
embedding = umaps.transform(y_pred)
embedding = pd.DataFrame(embedding)
adata = sc.AnnData(X=y_pred.copy())

adata.obsm['X_umap']=embedding.iloc[:,:2].values
adata.obs['cell_type']=y1.tolist()+y2.tolist()
sc.pl.umap(adata, color='cell_type', title='',size=75)
adata.obs['Source']=np.repeat(['Domain1','Domain2'],repeats=[len(y1),len(y2)])
sc.pl.umap(adata, color='Source', title='',size=75)

##-----------------------------------------------------------------------------
##diagonal integration via cross-modal autoencoder
from network import *
from crossmodalAE import *

import torch
import torch.nn.functional as F

FeatureExtractorA, FeatureExtractorB= crossmodalAE(epoch=100, batch_size=128,
                                                   source_trainset=domain1,
                                                   target_trainset=domain2)

X_all_tensor_a = torch.tensor(domain1).float()
X_all_tensor_b = torch.tensor(domain2).float()
FeatureExtractorA.to(torch.device("cpu"))
FeatureExtractorB.to(torch.device("cpu"))
y_preda = FeatureExtractorA(X_all_tensor_a)
y_preda = F.normalize(y_preda, dim=1,p=2)
y_predb = FeatureExtractorB(X_all_tensor_b)
y_predb = F.normalize(y_predb, dim=1,p=2)
y_pred = torch.cat((y_preda,y_predb),0)
y_pred = torch.Tensor.cpu(y_pred).detach().numpy()

##visualization
umaps = umap.UMAP(n_neighbors=15, min_dist=0.2, n_components=2,
                  metric="correlation").fit(y_pred)
embedding = umaps.transform(y_pred)
embedding = pd.DataFrame(embedding)
adata = sc.AnnData(X=y_pred.copy())

adata.obsm['X_umap']=embedding.iloc[:,:2].values
adata.obs['cell_type']=y1.tolist()+y2.tolist()
sc.pl.umap(adata, color='cell_type', title='',size=75)
adata.obs['Source']=np.repeat(['Domain1','Domain2'],repeats=[len(y1),len(y2)])
sc.pl.umap(adata, color='Source', title='',size=75)

##-----------------------------------------------------------------------------
##diagonal integration via SCIM
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from itertools import cycle

from pathlib import Path

from scim.model import VAE, Integrator
from scim.discriminator import SpectralNormCritic
from scim.trainer import Trainer
from scim.utils import switch_obsm, make_network, plot_training, adata_to_pd
from scim.evaluate import score_divergence, extract_matched_labels
from scim.evaluate import get_accuracy, get_confusion_matrix, get_correlation
from scim.matching import get_cost_knn_graph, mcmf

##------------------------
##set up model
OUTDIR = Path('scim_training/')

SEED = 12
TECHS = ['domain1', 'domain2']

latent_dim = 8
label_cats = y1.cat.categories

discriminator_net = make_network(
    doutput=1,
    units=[8]*2,
    dinput=latent_dim + label_cats.size,
    batch_norm=False,
    name='discriminator')

discriminator = SpectralNormCritic(
    discriminator_net,
    input_cats=label_cats)

vae_lut = dict()
for tech in TECHS:
    encoder = make_network(
        doutput=2*latent_dim, 
        units=[32]*2,
        dinput=1200,
        batch_norm=True, dropout=0.2,
        name=f'{tech}-encoder')

    decoder = make_network(
        doutput=1200,
        units=[32]*2,
        dinput=latent_dim,
        batch_norm=True, dropout=0.2,
        name=f'{tech}-decoder')

    vae_lut[tech] = VAE(
        encoder_net=encoder,
        decoder_net=decoder,
        name=f'{tech}-vae')
    
genopts = {key: tf.keras.optimizers.Adam() for key in TECHS}
disopt = tf.keras.optimizers.Adam()

trainer = Trainer(
    vae_lut=vae_lut,
    discriminator=discriminator,
    source_key='domain1',
    disopt=disopt,
    genopt_lut=genopts,
    beta=0.5)

##------------------------
##data preparation
train = dict()
test = dict()
full = dict()

full['domain1']=sc.AnnData(X=domain1.copy(),obs=pd.DataFrame(y1).astype('category'))
full['domain1'].obs['cell']=full['domain1'].obs[0]
full['domain2']=sc.AnnData(X=domain2.copy(),obs=pd.DataFrame(y2).astype('category'))
full['domain2'].obs['cell']=full['domain2'].obs[0]
mask = np.random.choice([False, True], domain1.shape[0], p=[0.2, 0.8])
train['domain1'] = full['domain1'][mask]
test['domain1'] = full['domain1'][~mask]

mask = np.random.choice([False, True], domain2.shape[0], p=[0.2, 0.8])
train['domain2'] = full['domain2'][mask]
test['domain2'] = full['domain2'][~mask]

# Convert train into tf.data.Datasets
for tech, val in train.items():
    train[tech] = tf.data.Dataset.from_tensor_slices((val.X, val.obs['cell'].cat.codes, val.obs_names.astype(int)))
    train[tech] = train[tech].batch(128, drop_remainder=True)

##---------------------
##initialization
ckpt_dir = OUTDIR.joinpath('1_init', 'model')

source = 'domain1'
target = 'domain2'

print('Initializing latent space by training VAE')
np.random.seed(SEED)
tf.compat.v1.random.set_random_seed(SEED)
gs = 0
for epoch in range(15):
    print(epoch, gs)
    for (data, dlabel, _), (prior, plabel, _) in zip(train[source], cycle(train[target])):

        # Initializes latent space
        loss, (mse, kl), (codes, recon) = trainer.vae_step(source, data, beta=0.001)
        # Initialize the discriminator
        disc_loss, _ = trainer.discriminator_step(source, target, data, prior, dlabel, plabel)
        # Record
        if gs % 10 == 0:
            lut = {'loss': loss.numpy().mean(),
                   'mse': mse.numpy().mean(),
                   'kl': kl.numpy().mean()}
            trainer.record('vae', lut, step=gs)

        gs = gs + 1

trainer.saver.save(ckpt_dir.joinpath('ckpt'))
    
##------------------------------
##integration
ckpt_dir = OUTDIR.joinpath('2_integrate', 'model')

source = 'domain1'
target = 'domain2'

print('Training integration')
np.random.seed(SEED)
tf.compat.v1.random.set_random_seed(SEED)
gs = 0
for epoch in range(25):
    for (data, dlabel, didx), (prior, plabel, pidx) in zip(train[source], cycle(train[target])):

        # Train the discriminator
        for _ in range(trainer.niters_discriminator):
            disc_loss, _ = trainer.discriminator_step(source, target, data, prior, dlabel, plabel)
            
        # Fool adversary & reconstruct
        loss, (nll, adv), (codes, recon) = trainer.adversarial_step(source, data, dlabel)
        
        # Evaluate training batch, write to trainer.history
        if gs % 5 == 0:
            batch = full[source][didx.numpy()]
            trainer.forward(source, batch, 'cell')

            pbatch = full[target][pidx.numpy()]
            trainer.forward(target, pbatch, 'cell')

        # Evaluate testset
        if gs % 50 == 0:
            evald = trainer.evaluate(test, 'cell')
            probs = evald.obs.groupby('tech')['probs-discriminator'].mean()
            dloss = evald.obs.groupby('tech')['loss-discriminator'].mean()
            mse = evald.obs.groupby('tech')['loss-mse'].mean()
            lut = {'divergence': evald.uns['divergence']}
            lut.update({f'probs-{k}': v for k, v in probs.to_dict().items()})
            lut.update({f'mse-{k}': v for k, v in mse.to_dict().items()})
            lut.update({f'discriminator-{k}': v for k, v in dloss.to_dict().items()})

            trainer.record('test', lut, gs)

        gs = gs + 1

trainer.saver.save(ckpt_dir.joinpath('ckpt'))
        
evald = trainer.evaluate(test, 'cell')
sc.tl.pca(evald)
sc.pp.neighbors(evald, n_neighbors=15, n_pcs=30)
sc.tl.umap(evald, min_dist=0.2)
sc.pl.umap(evald,color=['cell','tech'])