import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def analyse_weights(model):
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

def analyse_activations(model, input_tensor):
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    for name, layer in model.named_modules():
        layer.register_forward_hook(get_activation(name))

    with torch.no_grad():
        _ = model(input_tensor)

    for name, activation in activations.items():
        print(f"Activation: {name} | Size: {activation.size()} | Values: {activation[:2]} \n")

if __name__ == "__main__":
    config_path = r"C:\Users\suyas\VGG\LDM\latent-diffusion\configs\latent-diffusion\celebahq-ldm-vq-4.yaml"
    ckpt_path = r"C:\Users\suyas\VGG\LDM\latent-diffusion\models\first_stage_models\vq-f4\model.ckpt"
    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, ckpt_path, verbose=True)

    # Analyse weights
    analyse_weights(model)

    # Analyse activations
    input_tensor = torch.randn(1, 3, 256, 256).cuda()  # Example input tensor
    analyse_activations(model, input_tensor)