<h1 style="text-align:center">Logo Generation with Diffusion Models</h1>
<p style="text-align:center; font-size:15px">Y.Benjelloun, A.Bruez, N.Chek, H.Talaoubrid</p>

<br>
<img src="img/logos.png" style="display:block; max-width:500px; margin-left:auto; margin-right:auto"></img>
<br>

## Table of contents
- [Motivation](#motivation)
- [Diffusion models](#diffusion-models)
- [Theory : finetuning DALL-E 2](#theory--finetuning-dall-e-2)
- [Collecting data](#collecting-data)
- [Training CLIP](#training-clip)
- [Results](#results)
- [Introspection](#introspection)

## Motivation 

Creating logos for both commercial and artistic purposes is a complex task, historically assigned to humans. We are seeing these on a daily-basis : in the streets, in public transports, in TV advertisement and in almost every man-made objects.

We then can easily notice some usual patterns in logo design (shape, color, text...) related to the activity the logo is supposed to stand for. 

As we are aware of how competitive new **Diffusion Models** are, we wondered if it was possible to use them for this purpose. 

This could help :

* Save **time**
* Save **money**
* Be **accessible** (no design competences required)

## Diffusion models

Diffusion models are text-to-image Machine Learning models. 

We collected and tried few of them (locally) in order to get more confident with the way they work. It enabled us to get used to their associated open-source projects. 

Their uses require some resources such as important GPU VRAM. Some of them can be ran on CPU but with lower performances and longer runtime.

This is some of the results we get :

* **Stable diffusion [using CompVis project](https://github.com/CompVis/stable-diffusion)** : 

| Text         | Image |
|--------------|:-----:|
| A bald guy skiing in a green plain |  <img src="img/A_bald_guy_skiing_in_a_green_plain.png" style="display:block; max-width:30px; margin-left:auto; margin-right:auto"></img> |
| A fireman saving a child from a burning castle |  <img src="img/A_fireman_saving_a_child_from_a_burning_castle.png" style="display:block; max-width:150px; margin-left:auto; margin-right:auto"></img> |
| A judo champion tanning on a beach |  <img src="img/A_judo_champion_tanning_on_a_beach.png" style="display:block; max-width:150px; margin-left:auto; margin-right:auto"></img> |
| A little smurf riding a dog |  <img src="img/A_little_smurf_riding_a_dog.png" style="display:block; max-width:150px; margin-left:auto; margin-right:auto"></img> |

<br>

* **DALLE Mini [using this project](https://github.com/borisdayma/dalle-mini)** : 

| Text         | Image |
|--------------|:-----:|
| Beautiful sunset on a lake |  <img src="img/dalle-mini-output-0.png" style="display:block; max-width:150px; margin-left:auto; margin-right:auto"></img> |
| The Eiffel tower on the night |  <img src="img/dalle-mini-output-1.png" style="display:block; max-width:150px; margin-left:auto; margin-right:auto"></img> |

<br>

* **DALLE-2 [using this project](https://github.com/LAION-AI/dalle2-laion)** : 

*This is not the official [OpenAI DALL-E 2](https://openai.com/dall-e-2/) version but a replica one trained on [LAION dataset](https://laion.ai/blog/laion-5b/).*

| Text         | Image |
|--------------|:-----:|
| Beautiful corgi playing soccer |  <img src="img/beautiful_corgi_playing_soccer.png" style="display:block; max-width:150px; margin-left:auto; margin-right:auto"></img> |

## Theory : finetuning DALL-E 2

## Collecting data

## Training CLIP

## Results

## Introspection