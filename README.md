# dnnls_final_project_2

This is my DNNL project. The task is: given 4 frames of a comic-style story (each with an image and
a text description), predict what the 5th frame looks like, both the image and the text. Module is
Deep Neural Networks and Learning Systems (55-710365), Sheffield Hallam.

## What the model actually is

A visual encoder (either a small CNN I built or a frozen pretrained ResNet-18, depending on the
experiment) turns each frame into a vector. A text encoder (pretrained LSTM autoencoder, kept frozen
in most experiments) does the same for the description. These get fused - either just concatenated
together, or run through a cross-modal attention block I wrote so the image and text vectors can
attend to each other before fusing. The fused sequence goes through a GRU to get one vector
representing "what should come next", which then gets decoded back into an image and some text.

Dataset is [daniel3303/StoryReasoning](https://huggingface.co/datasets/daniel3303/StoryReasoning) on
HuggingFace. Trained everything in Colab since I don't have a GPU locally.

## Results

Honestly the loss numbers across most experiments are annoyingly close together, which made it hard
to tell what was actually helping. Here's what I got:

![Training loss across all 11 experiments](results/all_experiments_loss_comparison.png)
**Figure 1:** Training loss curves for all 11 experiments, plotted together. Most variants (concat
vs. cross-modal attention, CNN vs. ResNet-18, BiGRU, deeper GRU) sit in a flat cluster around
4.3-4.4 and barely move. Exp 8/9 (perceptual loss) start higher on a different loss scale but trend
down. The two curves that actually break away from the cluster and keep dropping - Exp 5 and Exp
10 - are the only ones with the text decoder unfrozen and 60 full epochs, ending at 3.61. That
combination is the one change that mattered; everything else is noise around the same plateau.

**Table 1:** Final training loss per experiment (same data as Figure 1, as numbers).

| # | What changed | Epochs | Final loss |
|---|---|---|---|
| 0 | Baseline (CNN + concat + GRU) | 25 | 4.361 |
| 1 | Cross-modal attention instead of concat | 25 | 4.360 |
| 2 | Swapped in frozen ResNet-18 | 35 | 4.406 |
| 3 | ResNet-18 + cross-modal attention together | 25 | 4.371 |
| 4 | Bigger latent dim (64 instead of 16) | 25 | 4.361 |
| 5 | Unfroze the text decoder + wider GRU (64) | 60 | **3.61** |
| 6 | Bidirectional GRU | 25 | 4.362 |
| 7 | 2-layer GRU | 25 | 4.352 |
| 8 | Added VGG perceptual loss | 25 | 4.31 |
| 9 | latent64 + perceptual loss + unfrozen decoder combined | 30 | 4.31 |
| 10 | Sharper decoder (pixel shuffle instead of transposed conv) | 60 | 3.61 |

Exp 5 is clearly the best number, but I should be upfront that it also just got way more epochs (60
vs 25-35 for everything else), so it's not really a fair one-variable-at-a-time comparison, more like
"the thing I happened to let train the longest also unfroze more of the model." Worth saying that
plainly rather than pretending it's a clean result.

Also, exp 8 and 9's numbers aren't really comparable to the others since they have an extra VGG
perceptual loss term added on top of the normal loss, so the number itself is measuring something
slightly different.

![Exp 5 loss curve](results/exp5/experiment_5_unfreeze_text_decoder_gru-64_(60_epochs)_loss.png)
**Figure 2:** Training loss for Exp 5 on its own (unfrozen text decoder + GRU-64), 60 epochs. Loss
drops steadily across the full run with no plateau or divergence, ending at 3.61 - the lowest of any
experiment.

## What the predictions actually look like

The loss table above hides something the numbers alone don't show: the predicted images across
every single experiment come out as a flat gray/beige blob, no visible structure. Here's exp 9 and
exp 10, predicted frame next to ground truth:

![Exp 9 prediction vs ground truth](results/exp9/exp9_prediction.png)
![Exp 10 prediction vs ground truth](results/exp10/exp10_prediction.png)
**Figure 3:** Predicted frame vs. ground truth, Exp 9 (top) and Exp 10 (bottom) - the two lowest-loss
runs after Exp 5. Both collapse to a near-uniform gray/beige field with no structure, despite falling
loss values throughout training. Text prediction and the loss curve were both improving normally, so
this is specifically an image-decoder failure - see the bug writeup below for the actual cause.

## Explainability (Grad-CAM)

Since this is a generation task rather than classification, there's no class score to backprop
from for Grad-CAM in the usual sense. Instead `src/gradcam.py` backprops from the image
reconstruction loss (L1 between predicted and target frame) on the last conv layer of the visual
encoder, using the exp 5 checkpoint (best loss, no retraining needed for this). The heatmap shows
which regions of an input frame the gradient says matter most for that loss.

![Grad-CAM example 1](results/exp5/gradcam_examples/gradcam_example_1.png)
![Grad-CAM example 2](results/exp5/gradcam_examples/gradcam_example_4.png)
**Figure 4:** Grad-CAM overlays on two validation frames, Exp 5 checkpoint. Warm regions (red/yellow)
mark where the gradient of the image reconstruction loss concentrates most. Across every example I
ran, that's consistently people/faces rather than background - a sane thing for the visual encoder
to prioritise, even though the frame it goes on to predict is the flat blob shown in Figure 3. More
examples are in `results/exp5/gradcam_examples/`.

## A bug I found (and only half-fixed)

At some point I actually looked at what the predicted images looked like instead of just trusting the
loss number, and they were basically a flat gray blob - no visible structure at all, even after the
full 60 epochs. Turned out the
decoder's forward() function was calling the same internal function twice and returning it as both
"content" and "context" - so it was literally the same tensor being pushed toward two different
targets at once (the real frame, and the average of the input frames) by two different loss terms.
That's almost certainly why the images across every experiment came out blurry - not just exp 10.

I fixed it by giving the decoder two separate output heads instead of one shared one. I didn't touch
the original decoder classes though, since experiments 0-10 above were already trained with the buggy
version and I didn't want to make that code silently not match what actually produced those results
anymore - the fix lives in a new class instead. Then I found a second problem on top: the text loss
number is just naturally way bigger than the image loss number (cross-entropy over a huge vocab vs.
L1 pixel loss), so even with two separate heads, the combined loss was still mostly "about text",
starving the image branch. Added a weight to fix that too.

Ran out of time to actually confirm this two-part fix works before the deadline. So this needs to be
said honestly in the presentation: found a real bug, understood why it was happening, fixed the code,
but couldn't finish verifying it before submission.

## Repo layout
- `src/model.py` - every architecture variant, roughly one comment block per experiment
- `src/train.py` - the training loop
- `src/utils.py` - dataset loading, CoT grounding stuff, the validation/plotting function
- `experiments.ipynb` - one section per experiment, meant to be run in Colab top to bottom
- `results/expN/` - trained weights + loss curve (and a prediction image, for some) per experiment
- `Baseline.ipynb` - the module's starter notebook, untouched
