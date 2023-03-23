
import numpy as np
def classify(model,data):
    label_decoder={0:'Flight will not be cancelled',1: 'Flight will be cancelled'}
    pred=model.predict(data)
    predi=label_decoder.get(pred[0])
    proba=model.predict_proba(data)
    proba_list=proba.tolist()[0]
    pred=np.argmax(proba_list)
    pred_prob=round(proba_list[pred],2)
    return predi,pred_prob