import numpy as np
from cca_zoo.deep.data import get_dataloaders
import lightning.pytorch as pl
from cca_zoo.deep import (
    DCCAE,
    DVCCA,
    SplitAE,
    architectures,
)
from cca_zoo.deep.data import NumpyDataset, check_dataset, get_dataloaders

import torch
import logging

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

def Deep_Models(X,Y, method = "DVCCA", LATENT_DIMS = 2, EPOCHS = 100, lr = 0.001, dropout = 0.05, nw=0):
    # pl.seed_everything(1)
    layer_sizes = (1024, 1024, 1024)
    K1 = X.shape[1]
    K2 = Y.shape[1]

    LATENT_DIMS = int(LATENT_DIMS)
    EPOCHS = int(EPOCHS)
    nw = int(nw)

    custom_dataset = NumpyDataset((X, Y))
    train_loader = get_dataloaders(custom_dataset, num_workers=nw)

    if method == "DVCCA":
        encoder_1 = architectures.Encoder(
            latent_dimensions=LATENT_DIMS,
            feature_size=K1,
            variational=True,
            layer_sizes=layer_sizes,
            dropout=dropout,
        )
        encoder_2= architectures.Encoder(
            latent_dimensions=LATENT_DIMS,
            feature_size=K2,
            variational=True,
            layer_sizes=layer_sizes,
            dropout=dropout,
        )
        decoder_1 = architectures.Decoder(
            latent_dimensions=LATENT_DIMS,
            feature_size=K1,
            layer_sizes=layer_sizes,
            activation = torch.nn.Identity(),
            dropout=dropout,
        )
        decoder_2 = architectures.Decoder(
            latent_dimensions=LATENT_DIMS,
            feature_size=K2,
            layer_sizes=layer_sizes,
            activation = torch.nn.Identity(),
            dropout=dropout,
        )
        dvcca = DVCCA(
            latent_dimensions=LATENT_DIMS,
            encoders=[encoder_1, encoder_2],
            decoders=[decoder_1, decoder_2],
            lr=lr,
        )
        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            enable_checkpointing=False,
            logger=False,
            deterministic=True,
            enable_progress_bar=False,
        )
        trainer.fit(dvcca, train_loader)
        recons = dvcca.recon(train_loader, mle=False)
        return recons
    if method == "DVCCAP":
        encoder_1 = architectures.Encoder(
            latent_dimensions=LATENT_DIMS,
            feature_size=K1,
            variational=True,
            layer_sizes=layer_sizes,
            dropout=dropout,
        )
        encoder_2= architectures.Encoder(
            latent_dimensions=LATENT_DIMS,
            feature_size=K2,
            variational=True,
            layer_sizes=layer_sizes,
            dropout=dropout,
        )
        private_encoder_1 = architectures.Encoder(
            latent_dimensions=LATENT_DIMS,
            feature_size=K1,
            variational=True,
            layer_sizes=layer_sizes,
            dropout=dropout,
        )
        private_encoder_2 = architectures.Encoder(
            latent_dimensions=LATENT_DIMS,
            feature_size=K2,
            variational=True,
            layer_sizes=layer_sizes,
            dropout=dropout,
        )
        private_decoder_1 = architectures.Decoder(
            latent_dimensions=2 * LATENT_DIMS,
            feature_size=K1,
            layer_sizes=layer_sizes,
            activation = torch.nn.Identity(),
            dropout=dropout,
        )
        private_decoder_2 = architectures.Decoder(
            latent_dimensions=2 * LATENT_DIMS,
            feature_size=K2,
            layer_sizes=layer_sizes,
            activation = torch.nn.Identity(),
            dropout=dropout,
        )
        dvccap = DVCCA(
            latent_dimensions=LATENT_DIMS,
            encoders=[encoder_1,encoder_2],
            decoders=[private_decoder_1, private_decoder_2],
            private_encoders=[private_encoder_1, private_encoder_2],
            lr=lr,
        )
        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            enable_checkpointing=False,
            logger=False,
            deterministic=True,
            enable_progress_bar=False,
        )
        trainer.fit(dvccap, train_loader)
        recons = dvccap.recon(train_loader, mle=True)
        return recons
    if method == "DCCAE":
        # X = X.reshape(-1)
        # Y = Y.reshape(-1)
        # class CustomDataset(torch.utils.data.Dataset):
        #     def __init__(self):
        #         pass

        #     def __len__(self):
        #         return 10

        #     def __getitem__(self, index):
        #         return {"views": (torch.from_numpy(X),torch.from_numpy(Y))}

        # custom_dataset = CustomDataset()
        # train_loader = get_dataloaders(custom_dataset)
        
        encoder_1 = architectures.Encoder(
            latent_dimensions=LATENT_DIMS, feature_size=K1, layer_sizes=layer_sizes
        )
        encoder_2 = architectures.Encoder(
            latent_dimensions=LATENT_DIMS, feature_size=K2, layer_sizes=layer_sizes
        )
        decoder_1 = architectures.Decoder(
            latent_dimensions=LATENT_DIMS,
            feature_size=K1,
            layer_sizes=layer_sizes,
            activation = torch.nn.Identity(),
            dropout=dropout,
        )
        decoder_2 = architectures.Decoder(
            latent_dimensions=LATENT_DIMS,
            feature_size=K2,
            layer_sizes=layer_sizes,
            activation = torch.nn.Identity(),
            dropout=dropout,
        )
        dccae = DCCAE(
            latent_dimensions=LATENT_DIMS,
            encoders=[encoder_1, encoder_2],
            decoders=[decoder_1, decoder_2],
            lr=lr,
            lam=0.5,
            optimizer="adam",
        )
        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            enable_checkpointing=False,
            logger=False,
            deterministic=True,
            enable_progress_bar=False,
        )
        trainer.fit(dccae, train_loader)
        recons = dccae.recon(train_loader, mle=True)
        return recons
    if method == "SplitAE":
        encoder_1 = architectures.Encoder(
            latent_dimensions=LATENT_DIMS, 
            feature_size=K1, 
            layer_sizes=layer_sizes
        )
        decoder_1 = architectures.Decoder(
            latent_dimensions=LATENT_DIMS,
            feature_size=K1,
            layer_sizes=layer_sizes,
            activation = torch.nn.Identity(),
            dropout=dropout,
        )
        decoder_2 = architectures.Decoder(
            latent_dimensions=LATENT_DIMS,
            feature_size=K2,
            layer_sizes=layer_sizes,
            activation = torch.nn.Identity(),
            dropout=dropout,
        )
        splitae = SplitAE(
            latent_dimensions=LATENT_DIMS,
            encoder=encoder_1,
            decoders=[decoder_1, decoder_2],
            lr=lr,
            optimizer="adam",
        )
        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            enable_checkpointing=False,
            logger=False,
            deterministic=True,
            enable_progress_bar=False,
        )
        trainer.fit(splitae, train_loader)
        recons = splitae.recon(train_loader, mle=True)
        return recons
