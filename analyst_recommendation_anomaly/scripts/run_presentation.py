"""Execute the presentation notebook and export an HTML report with embedded plots.

From the workspace: python3 analyst_recommendation_anomaly/scripts/run_presentation.py --mode demo
Install requirements-notebook.txt in this Python environment before running.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', type=Path, help='Notebook to execute; defaults to analyst_revision_presentation.ipynb')
    parser.add_argument('--mode', choices=['demo', 'real', 'existing'], default='demo')
    parser.add_argument('--extract', type=Path)
    parser.add_argument('--market-dir', type=Path)
    parser.add_argument('--run-dir', type=Path, help='Completed strategy run for existing mode')
    parser.add_argument('--start', default='2005-01-03')
    parser.add_argument('--end', default='2024-12-31')
    parser.add_argument('--strategy-python', type=Path, help='Interpreter used for the core backtest')
    parser.add_argument('--no-sensitivity', action='store_true')
    parser.add_argument('--show-code', action='store_true', help='Include input code in exported HTML')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--timeout', type=int, default=3600, help='Maximum seconds per notebook cell')
    args = parser.parse_args(argv)
    if args.mode == 'existing' and args.run_dir is None:
        parser.error('--mode existing requires --run-dir')
    if args.mode != 'existing' and args.run_dir:
        parser.error('--run-dir is only used with --mode existing')
    if args.mode != 'real' and (args.extract or args.market_dir):
        parser.error('--extract and --market-dir are only used with --mode real')
    try:
        import nbformat
        from nbclient import NotebookClient
        from nbconvert import HTMLExporter
    except ImportError as error:
        parser.error(f'{error}. Install requirements-notebook.txt using {sys.executable} -m pip install.')
    project = Path(__file__).resolve().parents[1]
    root = args.output_dir.resolve() if args.output_dir else project / 'outputs' / 'presentations'
    root.mkdir(parents=True, exist_ok=True)
    report = Path(tempfile.mkdtemp(prefix=f'{args.mode}_', dir=root))
    notebook_path = args.notebook.expanduser().resolve() if args.notebook else project / 'notebooks' / 'analyst_revision_presentation.ipynb'
    notebook = nbformat.read(notebook_path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    def absolute(path):
        return str(path.expanduser().resolve()) if path is not None else None
    settings = {'RUN_MODE': args.mode, 'EXTRACT_DIR': absolute(args.extract),
                'MARKET_DIR': absolute(args.market_dir), 'EXISTING_RUN_DIR': absolute(args.run_dir),
                'START': args.start, 'END': args.end, 'DOWNLOAD_DAILY': False,
                'DOWNLOAD_START': '2000-01-01', 'RUN_SENSITIVITY': not args.no_sensitivity,
                'REPORT_DIR': str(report), 'STRATEGY_PYTHON': absolute(args.strategy_python),
                'CONFIG_OVERRIDES': {}}
    parameters = next(c for c in notebook.cells if 'parameters' in c.metadata.get('tags', []))
    parameters.source = '# Injected by run_presentation.py\n' + '\n'.join(f'{k} = {v!r}' for k,v in settings.items())
    execution = {'complete': False, 'mode': args.mode, 'source_notebook': str(notebook_path), 'settings': settings,
                 'started_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.executable}
    execution_path = report / 'notebook_execution.json'
    execution_path.write_text(json.dumps(execution, indent=2))
    # A local kernel spec fixes the notebook kernel to this Python interpreter.
    # This prevents accidentally selecting an unrelated globally installed kernel.
    with tempfile.TemporaryDirectory(prefix='presentation_kernel_') as kernel_home:
        kernel_spec = Path(kernel_home) / 'kernels' / 'presentation-python'
        kernel_spec.mkdir(parents=True)
        (kernel_spec / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Presentation Python', 'language': 'python'}))
        previous = os.environ.get('JUPYTER_PATH')
        os.environ['JUPYTER_PATH'] = kernel_home + (os.pathsep + previous if previous else '')
        notebook.metadata.kernelspec = {'name':'presentation-python','display_name':'Presentation Python','language':'python'}
        try:
            print(f'Executing {args.mode} notebook; report directory: {report}', flush=True)
            client = NotebookClient(notebook, timeout=args.timeout, kernel_name='presentation-python',
                                    resources={'metadata': {'path': str(project)}}, record_timing=True)
            client.execute()
            # Portable metadata after the temporary kernel spec is removed.
            notebook.metadata.kernelspec = {'name':'python3','display_name':'Python 3','language':'python'}
            nbformat.write(notebook, report / 'analyst_revision_presentation.executed.ipynb')
            exporter = HTMLExporter(template_name='lab')
            exporter.exclude_input = not args.show_code
            exporter.exclude_input_prompt = True
            exporter.exclude_output_prompt = True
            html, _ = exporter.from_notebook_node(notebook)
            (report / 'analyst_revision_presentation.html').write_text(html)
            execution['complete'] = True
        except Exception as error:
            execution['error'] = str(error)
            nbformat.write(notebook, report / 'analyst_revision_presentation.failed.ipynb')
            raise
        finally:
            execution_path.write_text(json.dumps(execution, indent=2))
            if previous is None:
                os.environ.pop('JUPYTER_PATH', None)
            else:
                os.environ['JUPYTER_PATH'] = previous
    print(f'HTML presentation: {report / "analyst_revision_presentation.html"}')
    print(f'Executed notebook: {report / "analyst_revision_presentation.executed.ipynb"}')
    return report


if __name__ == '__main__':
    main()
