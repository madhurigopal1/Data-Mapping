"""
FastAPI dashboard app that exposes endpoints to run algorithm examples and view results.
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import importlib
import pkgutil

app = FastAPI(title="MLOps Dashboard")
app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")

# discover algorithms in ml_algorithms package
import ml_algorithms

ALGORITHMS = [mod.name for mod in pkgutil.iter_modules(ml_algorithms.__path__)]

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    html = ("""
    <!doctype html>
    <html>
      <head><title>MLOps Dashboard</title></head>
      <body>
        <h1>MLOps Dashboard - Prototype</h1>
        <div id="content"></div>
        <script>
          async function load(){
            const res = await fetch('/api/algorithms');
            const algs = await res.json();
            let html = '<ul>';
            algs.forEach(a=>{
              html += `<li>${a} <button onclick="runAlg(\'${a}\')">Run</button> <pre id=\'res_${a}\'></pre></li>`
            });
            html += '</ul>';
            document.getElementById('content').innerHTML = html;
          }
          async function runAlg(name){
            const res = await fetch(`/api/run/${name}`, {method: 'POST'});
            const data = await res.json();
            document.getElementById('res_'+name).innerText = JSON.stringify(data, null, 2);
          }
          load();
        </script>
      </body>
    </html>
    """)
    return html

@app.get('/api/algorithms')
async def list_algorithms():
    return ALGORITHMS

@app.post('/api/run/{algorithm}')
async def run_algorithm(algorithm: str):
    if algorithm not in ALGORITHMS:
        raise HTTPException(status_code=404, detail='Algorithm not found')
    # dynamic import and run train_and_eval
    try:
        mod = importlib.import_module(f'ml_algorithms.{algorithm}')
        if hasattr(mod, 'train_and_eval'):
            metrics = mod.train_and_eval()
            return {'algorithm': algorithm, 'metrics': metrics}
        else:
            return {'algorithm': algorithm, 'error': 'train_and_eval function not found'}
    except Exception as e:
        return {'algorithm': algorithm, 'error': str(e)}
