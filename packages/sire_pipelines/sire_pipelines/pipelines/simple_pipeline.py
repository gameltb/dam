import sire

from ..models.unet import UNet
from ..models.vae import VAE


class SimplePipeline:
    def __init__(self):
        # Hold strong references to the models
        self._unet = UNet()
        self._vae = VAE()
        # Create managed wrappers for them
        self.unet_managed = sire.manage(self._unet)
        self.vae_managed = sire.manage(self._vae)

    def __call__(self, x_vae, x_unet):
        print("Running VAE...")
        with sire.auto_manage(self.vae_managed) as vae_wrapper:
            vae_device = vae_wrapper.get_execution_device()
            print(f"  VAE is on device: {vae_device}")
            x_on_vae_device = x_vae.to(vae_device)
            x = self._vae(x_on_vae_device)
            print(f"  VAE output is on device: {x.device}")

        print("\nRunning UNet...")
        with sire.auto_manage(self.unet_managed) as unet_wrapper:
            unet_device = unet_wrapper.get_execution_device()
            print(f"  UNet is on device: {unet_device}")
            x_on_unet_device = x_unet.to(unet_device)
            x = self._unet(x_on_unet_device)
            print(f"  UNet output is on device: {x.device}")

        print("\nPipeline finished.")
        return x
