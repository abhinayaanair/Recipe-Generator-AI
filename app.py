from flask import Flask, request, render_template
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Initialize the Flask app
app = Flask(__name__)

# Load the model and tokenizer
checkpoint = "./results/checkpoint-2061"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device) 
# Define the route for the home page (frontend)
@app.route('/', methods=['GET', 'POST'])
def index():
    generated_recipe = None
    if request.method == 'POST':
        # Get user inputs from the form
        ingredients = request.form['ingredients']
        diet = request.form['diet']
        time = request.form['time']
        cuisine = request.form['cuisine']
        
        # Format the input text for the model
        input_text = f"Ingredients: {ingredients}; Time: {time}; Cuisine: {cuisine}; Diet: {diet}"
        
        # Tokenize the input text
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to("cuda")
        
        # Generate the recipe using the model
        outputs = model.generate(
        inputs["input_ids"],
        max_length=700,  # Ensure the output is long enough to capture full recipes
        num_beams=3,  # Beam search for better exploration
        temperature=0.6,  # Control randomness
        top_k=50,  # Limit search to top k tokens
        top_p=0.9,  # Nucleus sampling
        repetition_penalty=2.5,  # Penalize repetition
        early_stopping=True,
    )
        
        # Decode the generated output
        generated_recipe = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    return render_template('index.html', recipe=generated_recipe)

if __name__ == '__main__':
    app.run(debug=True)
