# GAN with Elastic Deformation

**Description:**  
This project applies the Elastic Deformation technique to the training data of a GAN model by inserting MNIST data with elastic transformations. The goal is to generate diverse images during the image generation process.

**Data:** MNIST

**Elastic Deformation :**
```
def elastic_deformation(image, alpha, sigma):
    random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    distorted_image = map_coordinates(image.reshape(shape), indices, order=1, mode='reflect')
    return distorted_image.reshape(shape)
```

**Elastic Deformation MNist image:**
![gan1](https://github.com/Masterjun12/Generative_model/blob/3695da09fac65eed1b0a90ee7071a1b02f01bed6/png/gan1.png)

---

# Variational AutoEncoders using abReLU

**Description:**  
In this project, we experiment with applying a new activation function with learnable parameters, alpha and beta, in place of the traditional ReLU function in Variational AutoEncoders.

**Data:** MNIST

**abReLU Function:**
```
class abReLU(nn.Module):
    def __init__(self, init_a=7.0, init_b=0.0):
        super(abReLU, self).__init__()

        self.a = nn.Parameter(torch.tensor(init_a, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(init_b, dtype=torch.float32))

    def forward(self, x):
        return torch.where(x >= self.b, self.a * (x - self.b), torch.zeros_like(x))
```

**Visual:**
![vae1](https://github.com/Masterjun12/Generative_model/blob/3695da09fac65eed1b0a90ee7071a1b02f01bed6/png/vae1.png)
![vae2](https://github.com/Masterjun12/Generative_model/blob/3695da09fac65eed1b0a90ee7071a1b02f01bed6/png/vae2.png)

---

# Denoising Diffusion Probabilistic Models with CIFAR-10

**Description:**  
This project focuses on studying Denoising Diffusion Probabilistic Models (DDPM) using CIFAR-10 data.

**Data:** CIFAR-10

**Generated Results:**
![ddpm1](https://github.com/Masterjun12/Generative_model/blob/3695da09fac65eed1b0a90ee7071a1b02f01bed6/png/ddpm1.png)
```
