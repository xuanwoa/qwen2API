# HF Space Deployment Notes

This file is for local deployment reference only. Do not upload it to the public Space.

## Files to Upload

Upload only these files from this directory to the Hugging Face Space:

- `README.md`
- `Dockerfile`

## Space Settings

- Type: Public Docker Space
- App port: `7860`
- Persistent Storage: enabled

## Secrets

Set this Hugging Face Space secret:

- `ADMIN_KEY=<strong-random-value>`

Do not leave `ADMIN_KEY` as the application default.

## Variables

No Hugging Face Variables are required. The thin Dockerfile sets `PORT=7860`.

## Persistent Data

When Hugging Face Persistent Storage is mounted at `/data`, the Space Dockerfile creates:

- `/data/runtime-data`
- `/workspace/data -> /data/runtime-data`

The runtime keeps using `/workspace/data`, while the actual files are stored on persistent storage.

## Image Update Flow

1. Push changes to GitHub `main`.
2. GitHub Actions builds and pushes `ghcr.io/xuanwoa/service-runtime:latest`.
3. Manually rebuild the Hugging Face Space so Docker pulls the latest image.

A normal Space restart does not rebuild the Docker image and may keep the previous pulled layer.

## Privacy Notes

The public Space repository should only contain the thin `README.md` and `Dockerfile`. The WebUI and runtime behavior remain visible to visitors because the Space is public and the WebUI is intentionally retained.
