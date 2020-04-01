from deeppavlov import Chainer


class NLUHandler:

    def __init__(self, tokenizer, slot_filler, intent_classifier):
        self.tokenizer = tokenizer  # preprocessing
        self.slot_filler = slot_filler  # another unit of pipeline
        self.intent_classifier = intent_classifier  # another unit of pipeline
        self.intents = []
        if isinstance(self.intent_classifier, Chainer):
            self.intents = self.intent_classifier.get_main_component().classes  # upper-level model logic


    def nlu(self, text):
        tokens = self.tokenize_single_text_entry(text)

        slots = None
        if callable(self.slot_filler):
            slots = self.extract_slots_from_tokenized_text_entry(tokens)

        intents = []
        if callable(self.intent_classifier):
            intents = self.extract_intents_from_tokenized_text_entry(tokens)

        return slots, intents, tokens

    def extract_intents_from_tokenized_text_entry(self, tokens):
        intent_features = self.intent_classifier([' '.join(tokens)])[1][0]
        if self.debug:
            # todo log in intents extractor
            intent = self.intents[np.argmax(intent_features[0])]
            # log.debug(f"Predicted intent = `{intent}`")
        return intent_features

    def extract_slots_from_tokenized_text_entry(self, tokens):
        return self.slot_filler([tokens])[0]

    def tokenize_single_text_entry(self, x):
        return self.tokenizer([x.lower().strip()])[0]
