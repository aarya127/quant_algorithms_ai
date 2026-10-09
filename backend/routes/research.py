"""
routes/research.py — research papers, notebooks, algorithm source blueprint, extracted verbatim from app.py.
"""
import os
import sys
import json
import datetime

from flask import Blueprint, jsonify, request, send_file, Response

import tempfile
import subprocess
from services import yf

bp = Blueprint('research', __name__)

# backend/ directory (this file lives one level deeper than app.py did; moved
# code that used os.path.dirname(__file__) now uses _BACKEND instead)
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@bp.route('/api/research/<paper_name>')
def get_research_paper(paper_name):
    """
    Serve a research paper PDF: prebuilt by the Docker build, or compiled from
    LaTeX on demand when running locally.
    """
    try:
        # Map paper names to their .tex file paths
        paper_map = {
            'heston': '../algorithms/volatility_forecasting/research/heston_model/theory.tex',
            'sabr': '../algorithms/volatility_forecasting/research/sabr_model/theory.tex',
            'state_space': '../algorithms/market_making_algorithms/state_space_models/theory.tex',
            'market_microstructure': '../algorithms/execution_algorithms/market_microstructure/theory.tex',
            'macd': '../algorithms/macd_rsi/macd/macd_theory.tex',
            'rsi': '../algorithms/macd_rsi/rsi/rsi_theory.tex',
            'greeks': '../algorithms/greeks/theory.tex',
            'derivatives_volatility': '../algorithms/volatility_forecasting/research/derivatives_volatility/theory.tex',
            'advanced_trading': '../algorithms/execution_algorithms/advanced_trading/theory.tex',
        }
        
        if paper_name not in paper_map:
            return jsonify({
                'success': False,
                'error': 'Research paper not found'
            }), 404
        
        # Get the absolute path to the .tex file
        tex_file = os.path.join(_BACKEND, paper_map[paper_name])
        tex_dir = os.path.dirname(tex_file)
        tex_filename = os.path.basename(tex_file)
        
        if not os.path.exists(tex_file):
            return jsonify({
                'success': False,
                'error': 'LaTeX source file not found'
            }), 404

        # The Docker build compiles every paper next to its .tex (the runtime image
        # has no LaTeX); compiling below is only for local runs with pdflatex.
        prebuilt = tex_file[:-len('.tex')] + '.pdf'
        if os.path.exists(prebuilt):
            return send_file(prebuilt, mimetype='application/pdf',
                             download_name=f'{paper_name}_model.pdf')

        # Create a temporary directory for compilation
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy the .tex file to temp directory
            import shutil
            temp_tex = os.path.join(tmpdir, tex_filename)
            shutil.copy(tex_file, temp_tex)
            
            # Compile LaTeX to PDF using pdflatex
            # Run twice to resolve references
            for _ in range(2):
                result = subprocess.run(
                    ['pdflatex', '-interaction=nonstopmode', '-output-directory', tmpdir, temp_tex],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
            
            # Check if PDF was generated
            pdf_filename = tex_filename.replace('.tex', '.pdf')
            pdf_path = os.path.join(tmpdir, pdf_filename)
            
            if not os.path.exists(pdf_path):
                return jsonify({
                    'success': False,
                    'error': 'PDF compilation failed',
                    'log': result.stderr
                }), 500
            
            # Read the PDF and return it
            return send_file(
                pdf_path,
                mimetype='application/pdf',
                as_attachment=False,
                download_name=f'{paper_name}_model.pdf'
            )
    
    except subprocess.TimeoutExpired:
        return jsonify({
            'success': False,
            'error': 'LaTeX compilation timed out'
        }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error generating PDF: {str(e)}'
        }), 500

@bp.route('/api/research/diagnostics/notebook')
def get_diagnostics_notebook():
    """
    Serve the diagnostics notebook as HTML: prebuilt by the Docker build, or
    converted with jupyter on demand when running locally.
    """
    try:
        # Path to the notebook
        notebook_path = os.path.join(
            _BACKEND, 
            '../algorithms/volatility_forecasting/research/diagnostics.ipynb'
        )
        prebuilt = notebook_path[:-len('.ipynb')] + '.html'
        if os.path.exists(prebuilt):
            return send_file(prebuilt, mimetype='text/html')

        if not os.path.exists(notebook_path):
            return jsonify({
                'success': False,
                'error': 'Notebook not found'
            }), 404
        
        # Create temporary directory for conversion
        with tempfile.TemporaryDirectory() as tmpdir:
            output_html = os.path.join(tmpdir, 'diagnostics.html')
            
            # Convert notebook to HTML without code cells (--no-input)
            result = subprocess.run(
                ['jupyter', 'nbconvert', '--to', 'html', '--no-input', 
                 '--output', output_html, notebook_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                return jsonify({
                    'success': False,
                    'error': 'Notebook conversion failed',
                    'details': result.stderr
                }), 500
            
            # Serve the converted HTML
            return send_file(
                output_html,
                mimetype='text/html',
                as_attachment=False
            )
            
    except subprocess.TimeoutExpired:
        return jsonify({
            'success': False,
            'error': 'Conversion timed out'
        }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bp.route('/api/algorithm/<name>')
def get_algorithm_source(name):
    """
    Serve Python algorithm source code as plain text.
    """
    algo_map = {
        'sabr_pricer':           '../algorithms/volatility_forecasting/volatility_models/sabr_pricer.py',
        'sabr_calibration':      '../algorithms/volatility_forecasting/volatility_models/calibration/run_calibration.py',
        'signal_generator':      '../algorithms/volatility_forecasting/volatility_models/signals/signal_generator.py',
        'strategy_signals':      '../algorithms/volatility_forecasting/volatility_models/signals/strategy_signals.py',
        'backtest_engine':       '../algorithms/volatility_forecasting/backtest_engine/engine.py',
        'portfolio_constructor': '../algorithms/volatility_forecasting/portfolio_engine/portfolio_constructor.py',
        'macd_strategy':         '../algorithms/macd_rsi/prototype.py',
        'greeks_calculator':     '../algorithms/greeks/prototype.py',
        'arima':                 '../algorithms/machine_learning_algorithms/time_series_models/arima.py',
        'garch':                 '../algorithms/machine_learning_algorithms/time_series_models/garch.py',
    }

    if name not in algo_map:
        return jsonify({'success': False, 'error': 'Algorithm not found'}), 404

    file_path = os.path.join(_BACKEND, algo_map[name])
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'error': 'Source file not found'}), 404

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        return source, 200, {'Content-Type': 'text/plain; charset=utf-8'}
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
