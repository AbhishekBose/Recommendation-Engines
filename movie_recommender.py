import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

data=fetch_movielens(min_rating=4.0)


#weighted approximate rank pairwise
#warp creates recommendation by looking at existing user rating pairs
#and uses gradient descent for optimization
model=LightFM(loss='warp')

model.fit(data['train'],epochs=30)

def sample_recommendation(model,data,user_ids):
    
#    number of items in train data 
    n_users,n_items=data['train'].shape
    
    for user_id in user_ids:
        #movies they already like
        known_positives=data['item_labels'][data['train'].tocsr()[user_id].indices]
        #movies we want to predict for
        scores=model.predict(user_id,np.arange(n_items))
        #rank them in order of most liked to least
        top_items=data['item_labels'][np.argsort(-scores)]
        
        #print out results
        print("User %s" % user_id)
        print("known_positives:")
        for x in known_positives[:3]:
            print("         %s" % x)
            
        print("Recommended:")
        for x in top_items[:3]:
            print("         %s" % x)
            

sample_recommendation(model,data,[1,29,200,400,349,129,234,467])