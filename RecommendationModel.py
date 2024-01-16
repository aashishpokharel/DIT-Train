
from src.utils import CONFIG

from src.inference_deepctr import (
    load_mappings, 
    load_model,
    get_top_k_recommendations_exploitation_model,
    get_top_k_recommendations_exploration_model
    )

class RecommendationModel(object):
    
    def __init__(self):

        self.loaded = False


    def load(self):
       
        self.item_id_to_aisle_id, \
        self.item_id_to_department_id, \
        self.ext_itemid_to_int_itemid, \
        self.ext_aisle_to_int_aisle, \
        self.least_interacted_categories_to_product_id, \
        self.cat_id_to_all_products_id, \
        self.products_thompson_params = load_mappings()

        self.MODEL = load_model()
        self.loaded = True
        print("Loaded model")

    def predict(self, X,  feature_names=None):


        if not self.loaded:
            self.load()
        k_exploit = CONFIG['k_exploit']
        k_explore = CONFIG['k_explore']
        x =  X.tolist()

        X = {
        "user_id": x[0][0],
        "weekday": x[0][1],
        "hour": x[0][2],
        "cate_ids": x[0][-1]
        }

        cate_ids = [self.ext_aisle_to_int_aisle[cate] for cate in X["cate_ids"]]
        
        exploit_inference_items = [item for cate in cate_ids for item in self.cat_id_to_all_products_id[cate]]
        explore_inference_items = [item for cate in cate_ids for item in self.least_interacted_categories_to_product_id[cate]]

        top_k_items = {}
        top_k_items["exploitation"] = get_top_k_recommendations_exploitation_model(self.MODEL, 
                                                                                inference_items = exploit_inference_items, 
                                                                                user_id = X["user_id"], 
                                                                                weekday = X["weekday"], 
                                                                                hour = X["hour"], 
                                                                                k = k_exploit, 
                                                                                item_id_to_aisle_id = self.item_id_to_aisle_id, 
                                                                                item_id_to_department_id = self.item_id_to_department_id)
        
        top_k_items["exploration"] = get_top_k_recommendations_exploration_model(explore_inference_items, k_explore)
        
        
        int_itemid_to_ext_itemid = {v:k for k, v in self.ext_itemid_to_int_itemid.items()}
        top_k_items["exploitation"] = [int_itemid_to_ext_itemid[itm] for itm in top_k_items["exploitation"]]
    
        return str(top_k_items)

