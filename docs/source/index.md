% HEART-library documentation master file, created by
% sphinx-quickstart on Wed Mar 27 15:42:18 2024.
% You can adapt this file completely to your liking, but it should at least
% contain the root `toctree` directive.

::::{grid} 2

:::{grid-item}
:child-align: center
HEART-library Documentation
===========================
The **Hardened Extension of the Adversarial Robustness Toolkit (HEART)** is a modular open-source Python library that provides AI developers and researchers with Testing and Evaluation (T&E) tools to assess AI model performance under adversarial attacks and improve model resiliency.  
```{button-link} quick_start/install_heart.html
:color: primary
:shadow:
Get Started with HEART {octicon}`arrow-right`
```
:::

:::{grid-item}

```{image} _static/theme/SVG/Example-8.svg
:alt: Description of the image
```
:::
::::

<hr style="margin-bottom:60px;">

::::::{grid} 2

:::::{grid-item-card} Quick Start
::::{grid} 2
:::{grid-item}
:columns: 3
```{image} _static/theme/SVG/quick-start.svg
:alt: Description of the image
:class: only-light
:target: quick-start/install_heart.html
```
```{image} _static/theme/SVG/quick-start-dark.svg
:alt: Description of the image
:class: only-dark
:target: quick_start/install_heart.html
```
:::
:::{grid-item}
:columns: 9
Use these resources to get an introduction to HEART and install the library
:::
:::{grid-item}
:class: home-tile-link-list
:columns: 12
- [Introduction to HEART](quick_start/intro_heart.md)
- [Installation and Setup](quick_start/install_heart.md)

<!-- ```{button-link} https://example.com
See all {octicon}`arrow-right`
``` -->
:::
::::
:::::

:::::{grid-item-card} Tutorials
::::{grid} 2
:::{grid-item}
:columns: 3
```{image} _static/theme/SVG/tutorial.svg
:alt: Description of the image
:class: only-light
:target: tutorials/index.html
```
```{image} _static/theme/SVG/tutorial-dark.svg
:alt: Description of the image
:class: only-dark
:target: tutorials/index.html
```
:::
:::{grid-item}
:columns: 9
If you are new to AI Security, the tutorials will introduce you to key concepts and workflows
:::
:::{grid-item}
:class: home-tile-link-list
:columns: 12
- [Drone Object Detection](tutorials/drone_tutorial/index.md)

<!-- ```{button-link} tutorials/index.html
See all {octicon}`arrow-right`
``` -->
:::
::::
:::::

::::::

::::::{grid} 2

:::::{grid-item-card} How-to Guides
::::{grid} 2
:::{grid-item}
:columns: 3
```{image} _static/theme/SVG/How-to.svg
:alt: Description of the image
:class: only-light
:target: how_to_guides/index.html
```
```{image} _static/theme/SVG/How-to-dark.svg
:alt: Description of the image
:class: only-dark
:target: how_to_guides/index.html
```
:::
:::{grid-item}
:columns: 9
If you are familiar with AI Security and know what you want to do, the how-to guides will show you step-by-step how to do it with HEART tools
:::
:::{grid-item}
:class: home-tile-link-list
:columns: 12
- [How to Simulate White Box Attacks](how_to_guides/white_box.md)
- [How to Simulate Black Box Attacks](how_to_guides/black_box.md)
- [How to Simulate Auto Attacks](how_to_guides/auto_attacks.md)

<!-- ```{button-link} how_to_guides/index.html
See all {octicon}`arrow-right`
``` -->
:::
::::
:::::

:::::{grid-item-card} Explanations
::::{grid} 2
:::{grid-item}
:columns: 3
```{image} _static/theme/SVG/explanation.svg
:alt: Description of the image
:class: only-light
:target: explanations/index.html
```
```{image} _static/theme/SVG/explanation-dark.svg
:alt: Description of the image
:class: only-dark
:target: explanations/index.html
```
:::
:::{grid-item}
:columns: 9
The explanations provide in-depth descriptions of relevant technical concepts
:::
:::{grid-item}
:class: home-tile-link-list
:columns: 12
- [Overview - Creating Adversarial Patches](explanations/PatchDocumentation.md)

<!-- ```{button-link} explanations/index.html
See all {octicon}`arrow-right`
``` -->
:::
::::
:::::

::::::

::::::{grid} 2

:::::{grid-item-card} Reference Materials
::::{grid} 2
:::{grid-item}
:columns: 3
```{image} _static/theme/SVG/reference.svg
:alt: Description of the image
:class: only-light
:target: references/index.html
```
```{image} _static/theme/SVG/reference-dark.svg
:alt: Description of the image
:class: only-dark
:target: references/index.html
```
:::
:::{grid-item}
:columns: 9
Reference Materials provide further resources for your understanding of both AI Security concepts and HEART tools
:::
:::{grid-item}
:class: home-tile-link-list
:columns: 12
- [Attack Cards](reference_materials/attack_cards/index.md)
- [Evaluation Pathways](reference_materials/evaluation_pathways.md)
- [Modules](modules/index.md)
<!-- ```{button-link} reference_materials/index.html
See all {octicon}`arrow-right`
``` -->
:::
::::
:::::

:::::{grid-item-card} About HEART
::::{grid} 2
:::{grid-item}
:columns: 3
```{image} _static/theme/SVG/about.svg
:alt: Description of the image
:class: only-light
:target: about/index.html
```
```{image} _static/theme/SVG/about-dark.svg
:alt: Description of the image
:class: only-dark
:target: about/index.html
```
:::
:::{grid-item}
:columns: 9
Learn more about HEART, its background, mission, and contributors
:::
:::{grid-item}
:class: home-tile-link-list
:columns: 12

- [Contributors](about/contributors.md)
- [Legal](legal.md)


:::
::::
:::::


::::::


```{toctree}
:caption: 'Quick Start'
:maxdepth: 2
:hidden:

   Introduction to HEART <quick_start/intro_heart>
   Installation and Setup <quick_start/install_heart>
```

```{toctree}
:caption: 'Documentation'
:maxdepth: 3
:hidden:

   Tutorials <tutorials/index>
   How-to Guides <how_to_guides/index>
   Explanations <explanations/index>
   Reference Materials <reference_materials/index>
   Glossary <glossary/index>
```

```{toctree}
:caption: 'About HEART'
:maxdepth: 2
:hidden:

   Contributors <about/contributors>
   Legal <legal>
```
