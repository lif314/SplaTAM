# datasets

## return
```python
if self.load_embeddings:
    embedding = self.read_embedding_from_file(self.embedding_paths[index])
    return (
        color.to(self.device).type(self.dtype),
        depth.to(self.device).type(self.dtype),
        intrinsics.to(self.device).type(self.dtype),
        pose.to(self.device).type(self.dtype),
        embedding.to(self.device),  # Allow embedding to be another dtype
        # self.retained_inds[index].item(),
    )

return (
    color.to(self.device).type(self.dtype),
    depth.to(self.device).type(self.dtype),
    intrinsics.to(self.device).type(self.dtype),
    pose.to(self.device).type(self.dtype),
    # self.retained_inds[index].item(),
)
```
