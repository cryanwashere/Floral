from src.floral import datasets

flowers_ds = datasets.from_sorted_dir("/Users/cameronryan/Desktop/flower_ds")
flowers_ds.save("flowers")

#flowers_ds = datasets.from_saved_npy("flowers")




print(flowers_ds.labels)