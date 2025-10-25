
from argparse import Namespace

import pandas as pd
from model import SASRec
import numpy as np
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
import torch
import gradio as gr

def load_args_from_file(config_content):
    args = Namespace()
    
    # Define type conversions based on the original argparse types
    type_map = {
        'batch_size': int, 'lr': float, 'maxlen': int, 'hidden_units': int,
        'num_blocks': int, 'num_epochs': int, 'num_heads': int, 
        'dropout_rate': float, 'l2_emb': float, 'inference_only': lambda s: s.lower() == 'true',
        'norm_first': lambda s: s.lower() == 'true',
        'dataset': str, 'train_dir': str, 'device': str, 'state_dict_path': str,
        'usernum': int, 'itemnum': int
    }

    for line in config_content.strip().split('\n'):
        if not line or ',' not in line: continue
        key, value_str = line.split(',', 1)
        key = key.strip()
        value_str = value_str.strip()
        
        # Apply type conversion
        if key in type_map:
            try:
                if key == 'state_dict_path' and value_str.lower() == 'none':
                    value = None
                else:
                    value = type_map[key](value_str)
            except ValueError:
                value = value_str # Default to string if conversion fails
        else:
            value = value_str
            
        setattr(args, key, value)
    return args

def search_movie(query):
    if not query:
        return gr.Dropdown(choices=[], value=None, visible=True)

    best_matches = process.extractBests(
        query=query,
        choices=MOVIE_TITLES,
        limit=10,
        scorer=fuzz.ratio
    )
    
    titles = [match[0] for match in best_matches]

    return gr.Dropdown(choices=titles, value=None, visible=True, label="Select a Movie")

def update_movie_list(selected_title, selected_ids, selected_titles_display, output_history):
    if not selected_title or selected_title in selected_titles_display:
        return selected_ids, selected_titles_display, output_history, gr.Button(visible=len(selected_ids) >= 1)

    movie_id = TITLE_TO_ID.get(selected_title)
    if movie_id is not None:
        new_ids = selected_ids + [movie_id]
        new_titles_display = selected_titles_display + [selected_title]
        print('selected title:', selected_title)
        print('selected titles display:', selected_titles_display)
        print('new titles display:', new_titles_display)
        print('new ids:', new_ids)
        submit_btn = gr.Button(visible=len(new_ids) >= 1)
        
        display_str = "\n\n".join(
            [f"• {title}" for title in new_titles_display]
        )

        return new_ids, new_titles_display, display_str, submit_btn

    return selected_ids, selected_titles_display, output_history, gr.Button(visible=len(selected_ids) >= 1)


def clear_selections():
    return [], "", "", gr.Button(visible=False), None, gr.Markdown("Recommendations will appear here.")



args_path = '/Users/stefano/Documents/projects/Movie-Recommendation/results/ratings_5stars_default/args.txt'
model_path = '/Users/stefano/Documents/projects/Movie-Recommendation/results/ratings_5stars_default/SASRec.epoch=30.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth'

with open(args_path, 'r') as f: 
    config_content = f.read()

args = load_args_from_file(config_content)

model = SASRec(args.usernum, args.itemnum, args)
model.load_state_dict(torch.load(model_path, map_location=torch.device(args.device)))

movies = pd.read_csv('data/movies.csv')
MOVIE_TITLES = movies.title.tolist()
TITLE_TO_ID = dict(zip(movies.title, movies.movieId))
ID_TO_TITLE = dict(zip(movies.movieId, movies.title))

def predict_top_movies(user_sequence):

    user_sequence = np.array(user_sequence)
    print('User sequence for prediction:', user_sequence, type(user_sequence))

    model.eval()
    seq = np.zeros([args.maxlen], dtype=np.int32)
    idx = args.maxlen - 1

    for movie in reversed(user_sequence):
        seq[idx] = movie
        idx -= 1

    model.eval()
    with torch.no_grad():
        predictions = model.predict(seq.reshape(1, -1), torch.arange(args.itemnum))

    scores = predictions.squeeze(0).cpu().numpy()  # (itemnum,)
    top_items = np.argsort(-scores)[:10]    

    recommended_titles = [ID_TO_TITLE.get(item_id, f"ID {item_id} (Not Found)") for item_id in top_items]
    
    markdown_list = "\n".join(
        [f"{i+1}. **{title}**" for i, title in enumerate(recommended_titles)]
    )
    
    result_str = "## 🎉 Top 10 Recommended Movies:\n\n" + markdown_list
    
    return result_str

with gr.Blocks() as demo:



    selected_ids = gr.State(value=[])
    selected_titles_display = gr.State(value=[])

    gr.Markdown("# 🎥 SASRec Movie Recommendation Demo")
    gr.Markdown(
        "Start typing a movie you've watched. Select at least **5** movies to get a prediction. "
    )

    with gr.Row():
        search_input = gr.Textbox(
            label="Search for a movie you've watched:", 
            placeholder="e.g., Pulp Fiction", 
            scale=4
        )
        clear_button = gr.Button("Clear All Selections", size="lg", scale=1)


    movie_dropdown = gr.Dropdown(
        label="CLICK HERE TO SELECT", 
        choices=[], 
        visible=True,
        interactive=True
    )

    gr.Markdown("---")
    gr.Markdown("## Your Watch History (Min. 5 required)")
    
    output_history = gr.Markdown(
        label="Selected Movies", 
        value="No movies selected yet.", 
    )

    submit_btn = gr.Button("Get Top 10 Recommendations", visible=False, variant="primary")

    gr.Markdown("---")
    
    result_output = gr.Markdown(
        label="Recommendations", 
        value="Recommendations will appear here.",
    )

    search_input.change(
        search_movie, 
        inputs=[search_input], 
        outputs=[movie_dropdown]
    )

    movie_dropdown.select(
        update_movie_list,
        inputs=[movie_dropdown, selected_ids, selected_titles_display, output_history],
        outputs=[selected_ids, selected_titles_display, output_history, submit_btn]
    )

    # Submission: Run prediction
    submit_btn.click(
        predict_top_movies,
        inputs=[selected_ids],
        outputs=[result_output]
    )
    
    # # Clear: Reset states and UI
    # clear_button.click(
    #     clear_selections,
    #     inputs=[],
    #     outputs=[selected_ids, output_history, result_output, submit_btn, search_input, result_output]
    # )

    demo.launch()



