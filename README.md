# Movies recommendations

The initial code for the model and training pipeline is taken from [this repository](https://github.com/pmixer/SASRec.pytorch). 

I used the [movie lens](https://grouplens.org/datasets/movielens/) dataset. For each user I only kept the movies with rating higher than 4.5. Using the architecture described in [SASRec](https://arxiv.org/abs/1808.09781) I trained the transformer model to predict the next movie from the previous ones. My intuition is to extract the preference of the user (all the movies he previously liked) and learn to suggest base on those. 

I achieved NDCG@10 = 0.9557 on the test set. 
