# -*- coding: utf-8 -*-
"""

"""

import flair
model = flair.models.TextClassifier.load('en-sentiment')
#en-sentiment model is a distilBERT model fitted with a classification head that outputs two classes 
#- negative and positive.

#%%

text = """The key interest rate has been raised by 0.25 points by the Bank of Canada. Mortgage payments are going up again. 
This decision will further burden families and businesses that are already struggling with the inflationary crisis."""
# we are expecting a neagive sentiment here

print(text)
sentence = flair.data.Sentence(text)

#%%
model.predict(sentence)
print('{}, score is: {}'.format
      (sentence.get_labels()[0].value, sentence.get_labels()[0].score))



