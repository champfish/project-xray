Score: 5/8

Apologies for the delay in getting this feedback to you. This looks solid so far, but your goals could use a little restructuring. You have a week from when you receive this until you need to submit a revised version. Please feel free to reach out with specific questions you have about this feedback, and please find a time to discuss it with your project mentor.

Formatting issue: can you try to make sure that your goals show up one per line, rather than in a giant block of text?

Essential goals:
1.	I’m assuming you plan to train your GAN models from scratch because you don’t specify any pretrained models. That’s totally fine if so, but is that correct? If you run into any major issues getting an initial GAN working, it might be worthwhile to look into pretrained models as a place to help you get started.
2.	Please pick just one initial quantitative evaluation to focus on as an essential goal. I would say either FID (you can use this: https://pytorch.org/ignite/generated/ignite.metrics.FID.html) or the classifier analysis that you currently use as your first stretch goal (see below). I’m not sure I fully understand the KNN analysis you hope to do, but that could also work.

Desired goals:
1.	How will you add noise to the data? To every image that the discriminator is shown? Or just to the real images that the discriminator is given? I might consider this as just another hyperparameter to consider rather than a goal on its own.
2.	I agree that you want to spend a fair amount of time considering different hyperparameters, but can you provide some specifics here about which hyperparameters you think are likely to be the most important and how you might try searching for the best values?

Stretch goals:
1.	As mentioned above, I don’t think this first stretch goal of training classifiers should be a stretch goal. It’s going to be much easier to set up a binary classifier for pneumonia vs. no pneumonia than it is to get the GAN training well. Because you’re specifically looking at a case where training data is limited and the generated images serve an obvious purpose in terms of trying to improve classification accuracy, this seems like it would be a good Desired goal. Additional experiments into determining how and whether you can change the GAN to improve this pneumonia classifier’s accuracy is going to be hard and open-ended (aka, that’s a reasonable stretch goal), but just training the model for a comparison is something you should certainly be able to do.
2.	The thought behind these stretch goals is that are something you can focus on if your earlier goals go smoothly, and they should each take your whole group roughly a week of work to complete. You aren’t in any way committing to *finishing* these goals, but I want a sense of what you hope to focus on if, for example, your GAN trains relatively smoothly. Can you add one additional stretch goal?


Possible ideas for another stretch goal:
-	Try using pretrained models either as your classifier or your generator
-	Try to use explainability/visualization methods to understand what specifically about your classifier improves or worsens when training it on synthetic data
-	Especially if the synthetic data does not help the classifier, try to see how small of an original dataset you need before the GAN-generated synthetic data *does* help.
