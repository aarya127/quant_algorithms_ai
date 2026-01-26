# Research Papers - Dynamic LaTeX Compilation

This directory contains research papers that are **dynamically compiled** from LaTeX source files.

## How It Works

Instead of storing large PDF files, we:
1. Keep only the source `.tex` files in the `quant_research/` directory
2. Compile them **on-demand** when users click "View PDF"
3. Serve the freshly generated PDF directly to the browser

## Benefits

✅ **Saves Space**: No need to store PDFs (can be 500KB-2MB each)  
✅ **Always Up-to-Date**: Any edits to `.tex` files are immediately reflected  
✅ **Version Control Friendly**: LaTeX source is much easier to diff/merge  
✅ **Professional**: LaTeX produces publication-quality PDFs  

## API Endpoints

- `GET /api/research/heston` - Compiles and returns Heston model PDF
- `GET /api/research/sabr` - Compiles and returns SABR model PDF
- `GET /api/research/advanced_trading/markdown` - Returns markdown content

## Requirements

- **pdflatex** must be installed on the system
- LaTeX packages: amsmath, amssymb, amsthm, graphicx, hyperref

## Performance

First compilation: ~2-3 seconds (includes two LaTeX passes for references)  
Subsequent compilations: Same (always fresh compilation)

For production, you could implement caching to speed this up.

## Adding New Papers

1. Create your `.tex` file in `quant_research/`
2. Add the path mapping in `app.py` under `@app.route('/api/research/<paper_name>')`
3. Add the paper card in `templates/index.html` under the Research Library tab
4. That's it! No PDF storage needed.
