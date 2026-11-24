"""Single-upload dependency and callback regressions, plus real Blocks when available."""
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from wafer.demo_ui import ProductUI, bind_single_events, valid_examples, SINGLE_EMPTY


class Component:
    def __init__(self):
        self.events = []
    def change(self, fn, **kwargs):
        self.events.append((fn, kwargs))
    click = change


class Service:
    def __init__(self):
        self.paths = []
    def single(self, path):
        self.paths.append(path)
        return 'original', 'heat', 'overlay', 'classification / Top-3'


def png(path):
    Image.fromarray(np.full((16, 16), 160, dtype=np.uint8)).save(path)
    return str(path)


def test_upload_preview_detect_clear_dependency_flow(tmp_path):
    components = [Component() for _ in range(8)]
    upload, preview, button, clear, original, heat, overlay, result = components
    service = Service()
    gr = SimpleNamespace(Progress=lambda: lambda *a, **k: None)
    bind_single_events(gr, ProductUI(service), *components)
    preview_fn, preview_dep = upload.events[0]
    detect_fn, detect_dep = button.events[0]
    clear_fn, clear_dep = clear.events[0]
    assert preview_dep['inputs'] == detect_dep['inputs'] == [upload]
    assert upload not in preview_dep['outputs'] + detect_dep['outputs']
    assert preview_dep['concurrency_id'] == detect_dep['concurrency_id'] == clear_dep['concurrency_id']
    path = png(tmp_path / 'valid.png')
    assert preview_fn(path)[0].shape == (16, 16, 3)
    assert service.paths == []
    assert detect_fn(path) == ('original', 'heat', 'overlay', 'classification / Top-3')
    assert service.paths == [path]
    cleared = clear_fn()
    assert cleared == (None, None, None, None, None, SINGLE_EMPTY)
    assert preview_fn(cleared[0]) == (None, None, None, None, SINGLE_EMPTY)
    assert '请先上传' in detect_fn(None)[-1]
    assert service.paths == [path]


def test_examples_validate_rows_and_reject_bad_png(tmp_path, caplog):
    valid = png(tmp_path / 'valid.png')
    (tmp_path / 'bad.png').write_text('not a PNG')
    Image.fromarray(np.full((16, 16), 0, dtype=np.uint8)).save(tmp_path / 'wrong_encoding.png')
    assert valid_examples(tmp_path) == [[valid]]
    assert 'Skipping invalid demo example' in caplog.text
    assert valid_examples(tmp_path / 'missing') == []


def test_real_gradio_blocks_dependencies_and_examples(tmp_path):
    gr = pytest.importorskip('gradio', reason='No installed Gradio; run this integration test in server environment')
    path = png(tmp_path / 'valid.png')
    with gr.Blocks() as app:
        upload = gr.File(type='filepath', file_types=['.png'])
        preview = gr.Image()
        button, clear = gr.Button(), gr.Button()
        original, heat, overlay, result = gr.Image(), gr.Image(), gr.Image(), gr.Markdown()
        examples = gr.Examples(examples=valid_examples(tmp_path), inputs=[upload],
                               cache_examples=False, run_on_click=False)
        bind_single_events(gr, ProductUI(Service()), upload, preview, button, clear,
                           original, heat, overlay, result)
    deps = app.config['dependencies']
    for name in ('preview', 'run_single'):
        matches = [d for d in deps if d.get('api_name') == name]
        assert len(matches) == 1
        assert matches[0]['inputs'] == [upload._id]
        assert matches[0].get('trigger_after') is None
        assert upload._id not in matches[0]['outputs']
    assert examples.examples == [[path]]
    assert examples.cache_examples is False
    app.close()
