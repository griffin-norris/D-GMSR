
def make_gif(frame_folder):
    frames = [Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/*"+".png"))][::2]
    frame_one = frames[0]
    frame_one.save(frame_folder+"/"+ "animation.gif", format="GIF", 
                   append_images=frames, save_all=True, duration=0.1, loop=0)