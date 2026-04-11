config = {
    "model":{
        "main-name":"sd-v1-5-inpainting",
        "vae-name":"vae-ft-mse-840000-ema-pruned"
        
    },
    
    
    "training":{
        "settings":{ ## Training Settings ##
            "epochs": 30,
            "batch-size":1,
            "lr":1e-4,
            "rank":4,
            
        },
                
        "dataset":{ ## Dataset Settings ##
            "trigger":"mytasktype", # The trigger word for your dataset
            "category-info":"", # For example, "dog, football, running" etc
            "quality-info":"high quality, detailed", # Describe overall quality of dataset

            "path":"./data", # path to dataset 
        },
        
        "outputs":{ ## Output Settings ##
            "path":"./outputs"
        },

        
        "save_every":1

    }
}