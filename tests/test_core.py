import ast
import importlib.util
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from security import hash_password, verify_password

def detection_namespace():
    tree = ast.parse((ROOT / 'app.py').read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'process_detections_py')
    namespace = {'detection_buffers': {}}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'detector_test', 'exec'), namespace)
    return namespace

class CoreTests(unittest.TestCase):
    def test_persistent_detection_and_cooldown(self):
        f = detection_namespace()['process_detections_py']
        self.assertFalse(f(1, True, 10000))
        self.assertFalse(f(1, True, 11000))
        self.assertTrue(f(1, True, 13000))
        self.assertFalse(f(1, True, 14000))
        self.assertFalse(f(1, True, 17000))
        self.assertTrue(f(1, True, 18000))

    def test_sparse_and_absent_detection(self):
        f = detection_namespace()['process_detections_py']
        self.assertFalse(f(1, False, 10000))
        self.assertFalse(f(1, False, 12000))
        self.assertFalse(f(1, True, 14000))

    def test_cameras_are_independent(self):
        f = detection_namespace()['process_detections_py']
        self.assertFalse(f(1, True, 10000))
        self.assertFalse(f(2, False, 10000))
        self.assertTrue(f(1, True, 14000))
        self.assertFalse(f(2, False, 14000))

    def test_passwords_are_salted_and_verified(self):
        a, b = hash_password('example-password'), hash_password('example-password')
        self.assertNotEqual(a, b)
        self.assertTrue(verify_password('example-password', a))
        self.assertFalse(verify_password('wrong', a))
        self.assertFalse(verify_password('example-password', 'not-a-hash'))

    def test_label_conversion(self):
        spec = importlib.util.spec_from_file_location('converter', ROOT / 'tools/convert_labels.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            labels = pathlib.Path(directory) / 'labels'
            labels.mkdir()
            target = labels / 'sample.txt'
            target.write_text('2 0.5 0.5 0.1 0.2\n\n7 0.2 0.3 0.2 0.4\n', encoding='utf-8')
            module.main(directory)
            self.assertEqual(target.read_text(), '0 0.5 0.5 0.1 0.2\n0 0.2 0.3 0.2 0.4\n')

if __name__ == '__main__':
    unittest.main()
