# HEART's Foundation Attack Catalogue
__🎯 Purpose:__ 

This directory houses HEART's catalogue of foundation attacks. The catalogue currently supports Foundation Adversarial Patches. 

__💡 Motivation:__

T&E users may use the foundation attack assets to
- evaluate their own models without needing to launch/optimize an attack, which can be time and compute intensive
- optimize further on their own model, reducing the time to identify an effective attack

__🛠️ Usage:__

T&E users may use the `catalogue` functionality of HEART to create, optimize and search for Foundation Adversarial Patches. Currently this is an experimental feature. All execution of the catalogue must occur in the `/notebooks/experimental/` directory. 

There is a _Getting Started_ notebook in this directory demonstrating how to search for existing Foundation patches in the catalogue, filter them by T&E attributes, optimize them further on your own model and create your own Foundation Patch from scratch. 

__🤝 Contribution:__

Once a user has created their own adversarial patch for effective test and evaluation on their model, they may wish to contribute the patch to the HEART repository for sharing with the wider T&E community. The following steps can facilitate this.

1. Fork the heart-library repository
2. Create a branch that describes your patch
3. Create your patch using the `create_patch` workflow
4. Update your patch artifacts to be relative paths __‼️ important step ‼️__
   - Navigate to your `./patches/{experiment_id}/{run_id}` and open the `meta.yaml` file
   - In `artifact_uri`, you will need to remove everything before the `../../patches` substring
4. Commit and push your patch artifacts i.e. everything in the `./patches/{experiment_id}/{run_id}` directory
5. Open a PR for your branch into the heart-library repository
