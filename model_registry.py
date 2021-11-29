from util import dict_to_tuple
import seml

COLLECTION = '_model_registry'

class ModelRegistry:

    def __init__(self, collection_name=COLLECTION):
        self.database = seml.database
        self.collection_name = collection_name

    def __getitem__(self, cfg):
        """ Gets the path to a trained model with a given configuration set. """
        collection = self.database.get_collection(self.collection_name)
        for doc in collection.find():
            if doc['config'] == cfg:
                return doc['path']
        return None

        registry = {
            doc['config'] : doc['path'] for doc in collection.find()
        }
        return registry.get(cfg, None)

    def __setitem__(self, cfg, cptk_path):
        """ Sets the path of a trained model with a given configuration set. """
        collection = self.database.get_collection(self.collection_name)
        if collection.delete_many({'config' : cfg}).deleted_count > 0:
            print(f'Overwriting path for configuration.')
        collection.insert_one({
            'config' : cfg,
            'path' : cptk_path,
        })


if __name__ == '__main__':
    reg = ModelRegistry()
    print(reg[{'test' : {'foo' : True, 'bar' : [0, 1]}}])
    reg[{'test' : {'foo' : True, 'bar' : [0, 1]}}] = 'test.path'
    print(reg[{'test' : {'foo' : True, 'bar' : [0, 1]}}])
