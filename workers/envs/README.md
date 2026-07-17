# Conda environments

Use the shared environment first:

```text
<workspace>/env/moshimo-box-kyutech/app
```

Create another environment only when a model requires conflicting versions.
Keep all environments one level below the project environment root:

```text
<workspace>/env/moshimo-box-kyutech/
  app/
  generation/
  flux2-klein/
  echomimic-v3/
```

`generation` contains Fish Audio S2 Pro. FLUX.2 Klein uses `flux2-klein`
because its required Diffusers and Hugging Face Hub versions conflict with
Fish's pinned Transformers stack. EchoMimicV3 uses its own Python 3.10
environment. Register
each Python executable in the local model catalog. Environments are outside
Git.
