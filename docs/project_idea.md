# How can we generate variations of the ZebraLogic puzzle automatically?
The ZebraLogic puzzle is really interesting because it allows us to evaluate how
a language model reasons over difficulty-adaptable problems (depending on the
size). Also, it allows us to easily modify instances of the problem in order to
look for possible biases or reasoning flaws in LLMs (e.g. gender bias). For example:

1. Gender/cultural/ethnic bias: Use woman names which are assigned to non-stereotypical
occupations/hobbies and check if the model's reasoning is breaking because of it.
2. Symbolic testing: Replace entities with symbols such as "Person A has hobby C".
3. Positional bias: Swap the order in which the clues are provided in the prompt.
4. World grounding bias: Check whether a model's reasoning breaks when prompted
with entity relationships that are not realistic according to our world knowledge,
e.g., Brad Pitt is a singer, someones' favourite food is sand, ...
5. Completely made up entities (words made up from randomly generated letters).

---

## Hypothesis:

We can use knowledge graphs (KG) to help us generate samples of entity-relationships
that allow us create new instances of a ZebraLogic problem.

In order to achieve this, we should be able to do the following:
1. Extract the template for a given ZebraLogic problem (i.e. replace entities with
variables that can be filled with new information).
2. Generate a population of entities for each variable considered in the benchmark
(name, favorite color, favorite drink, favorite sport, ...)
3. Create a KG based on this information.
4. Generate triplets that align with the LLM biases (e.g. most probable entity
among the available ones and least probable).
5. Fill the template with the new triplets.

Extra: This method can only generate problems that are well-aligned with the LLM's
knowledge/biases and problems that are the most missaligned ones within the entities
considered in the KG. Can we find a way to establish specific relationships that
form known biases such as gender bias in the KG? 

---

# On epistemic humility

What if we feed an unsolvable problem by changing/removing one of the clues?

The most probable answer is the model still generates a (of course) wrong solution.
This is what we call a lack of epistemic humility, that is, the LLM is not capable of 
recognizing its own ignorace.

The interesting thing about teaching an LLM to admit ignorance is you first need
a whole evaluation to check what does this specific LLM know and what it doesn't
know and only then, you can pick those samples where it was wrong and then feed
them through RLVR where the correct answer would be "I don't know" or similar.

Now, the most common approach is just to retrain/finetune the model to actually
*know* what it didn't know during the evaluation, but that only leads to the
same overconfident LLM with tons of hallucinations that we are trying to avoid.

According to our paper **Exploring Linguistic Competence in Small Language Models**, 
this small models are not really good at dealing with factual knowledge tasks so 
we think trying to provide more factual information into these models through 
RL over the answers it was previously wrong does not really make any sense. They 
are just too small to *know* everything so we would really prefer them to be able
to self-recognize what they don't know and simply admit it.

This aligns with the set of experiments we have been working on previously (**EvaLex**,
the MCQA permutations, ...) as they all look for automate, label free ways of
evaluating LLMs to capture samples where we are certain the LLM just didn't know
the answer and then, we could train the model to answer "I don't know" to those 
same questions.