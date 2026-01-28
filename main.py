from diffusers import StableDiffusionPipeline
import torch
from PIL import Image,ImageEnhance,ImageFilter
pipe= StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)
pipe=pipe.to("cpu")

def generate_image(prompt):
    return pipe(prompt).images[0]

def post_processsing(image):
     enhancer=ImageEnhance.Brightness(image)
     image=enhancer.enhance(1.2)
     enhancer=ImageEnhance.Contrast(image)
     image=enhancer.enhance(1.3)
     image=image.filter(ImageFilter.GaussianBlur(2))
     return image

def main():
     print("Welcome to the Post-Processing Magic Workshop!")
     print("This program generates an image from text and applies post-processing effects")
     print("type 'exit' to quit")
     while True:
          prompt=input("enter a description for the image or (type 'exit' to quit): ")
          if prompt.lower()=='exit':
               print("Goodbye")
               break
          try:
               print("generating image...")

               img=generate_image(prompt)
               img=post_processsing(img)
               img.show()

               save_option=input("do you want to save the image (yes or no): ")
               if save_option.lower()=='yes':
                     file_name=input("enter name of file: ")
                     img.save(f"{file_name}.png")
                     print(f"Image saved as {file_name}.png")
               print("-"*80+"\n")
          except Exception as e:
               print(f"An error occured:{e}\n")

if __name__=="__main__":
     main()