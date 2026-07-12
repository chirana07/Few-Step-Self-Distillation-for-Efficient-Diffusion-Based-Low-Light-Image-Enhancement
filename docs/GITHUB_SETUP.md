# Creating the GitHub repository

Run these commands from this package directory:

```bash
cd GitHub_Release_Package
git init
git lfs install
git add .gitattributes .gitignore README.md requirements.txt docs scripts notebooks configs results figures checkpoints *.py
git status
git commit -m "Initial reproducibility package for LUMIDIFF FSD"
git branch -M main
git remote add origin https://github.com/<account>/<repository>.git
git push -u origin main
```

The six `.pth` files are configured for Git LFS. GitHub's normal Git storage
must not be used for them. If Git LFS quota is insufficient, attach the
checkpoint binaries to a GitHub Release and keep the SHA-256 registry in
`docs/CHECKPOINTS.md`.
