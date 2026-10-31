Figure SP. The whole accessibility-weight curve for every scored arm, behind the optima
counted in Figure SM. Rows are deep mutational scanning panels and columns the two models,
so each row is one paired comparison. The dashed grey line fits all replacements together
and the three coloured lines fit inside one nucleotide-step class each; a marker is that
curve's best weight. 18 arms over 9 panels.

Every fit is drawn. The two receptor-binding assays are shaded and labelled "not counted":
their correlations are negative across the whole sweep, because they measure a phenotype
running against fitness rather than with it, so the largest value falls wherever the model
is diluted hardest towards zero and no optimum is defined. Those panels are plotted
anyway, because the shape of their curves is what justifies leaving them out. Among the
assays that are counted, nothing is excluded on how good its fit looks; two kinds are
marked so that they can be discounted by the reader rather than by the figure. A best
weight at the end of the swept range is drawn as a square (3 fits): it is a bound rather
than a turning point, since the curve may still be improving past the edge, but where a
curve runs plainly towards that edge it is a result and not a failure. A curve whose
correlation never reaches 0.05 anywhere in the sweep is drawn hollow on a shaded panel (1
fit): its optimum may still be perfectly clear -- one such curve falls steadily across the
whole sweep and is read at the far edge -- but nothing there is strong enough to lean on.
Where such a curve is also unmoved by the weighting, gaining less than 0.0005 in rho over
alpha = 0, no marker is drawn (2 fits): the sweep found nothing there, so the position of
the largest value is numerical noise rather than an optimum. Those fits are still counted,
as showing no preference, which is what keeps the denominators equal across classes. The
receptor-binding panels carry no markers either (16 fits), for the different reason that
no optimum is defined for them at all.

A negative weight -- dividing some mutational accessibility back out -- is preferred in 11
of 14 pooled fits and 33 of 42 fits made inside a single nucleotide-step class (14 of 14
at 1, 12 of 14 at 2, 7 of 14 at 3). Every one of those denominators is the same because
every arm is counted in every class.

Those totals pool two models that behave differently, and in the pooled fit the two
separate completely. Every one of the 7 fine-tuned arms prefers a negative weight, between
alpha = -0.24 and -0.13 (median -0.19), while the base model's 7 arms sit either side of
zero, between -0.06 and +0.04 (median -0.03), only 4 of them negative. The two ranges do
not overlap: the least negative fine-tuned arm still asks for more accessibility to be
divided out than the most negative base arm does, so an arm's model can be read off its
preferred weight alone, and a single median over all 14 would fall in the empty gap
between the two groups and describe neither. This is what the paper's argument predicts:
dividing accessibility back out helps where the model absorbed it, and the fine-tuned
model is the one that did.

Inside a single nucleotide-step class that separation goes. At one nucleotide change both
models prefer a negative weight in all 7 of their arms, so the preference there is not a
fine-tuning effect. By three changes neither model shows one: 2 of 7 base and 5 of 7
fine-tuned arms are negative, over a spread wide enough to hold both signs. Weights are
not comparable between the pooled fit and a single class, nor between classes: fixing the
step count collapses the spread of log accessibility about fivefold, so the same real
down-weighting needs a correspondingly larger alpha inside a class. On the scale-matched
footing alpha x SD(log accessibility) the median is -0.36 pooled against -0.29 within
classes.
