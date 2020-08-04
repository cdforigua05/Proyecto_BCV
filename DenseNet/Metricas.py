import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

melanomas = ["Melanoma","Melanotyc Venus","Basal Cell Carcinoma","Actinic Keratosis","Benign Keratosis","Dermato fibroma"\
				,"Vascular Lesion","Squamous cell carcinoma"]

def F_Medida(target, pred): 
	pred = pred.data.max(1)[1]
	pred = pred.cpu()
	target = target.cpu()
	scores = f1_score(target,pred,average=None)
	return scores

def PR(target,pred): 
	pred = pred.cpu().detach().numpy()
	target = target.cpu().detach().numpy()
	target_bin = label_binarize(target,classes=[0,1,2,3,4,5,6,7])
	classes = target_bin.shape[1]
	AUC = {}
	
	for i in range(classes):
		AUC[i] = average_precision_score(target_bin[:,i],pred[:,i])
	
	return AUC

def F_score_PR(target,pred): 
	scores = F_Medida(target, pred)
	AUC = PR(target, pred)
	print("-"*60)
	print("Name".ljust(23),"|F-Score|AP")
	
	for name,f1,AP in zip(melanomas,scores,AUC.values()):
		print(name.ljust(23),"|{:.4f}".format(f1)+" |{:.4f}".format(AP))
	print("Mean".ljust(23),"|{:.4f}".format(np.mean(scores))+" |{:.4f}".format(np.mean(list(AUC.values()))))
	print("-"*60)
	return ( np.mean(scores), np.mean(list(AUC.values())) )

		

	



