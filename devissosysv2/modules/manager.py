# Module Manager
# Modülleri yöneten ana sınıf

import importlib
from typing import Dict, Any, List
from .config import config

class ModuleManager:
    """Modül yöneticisi sınıfı"""
    
    def __init__(self):
        self.loaded_modules = {}
        self.module_instances = {}
    
    def load_module(self, module_name: str):
        """Modülü yükle"""
        try:
            if not config.is_module_enabled(module_name):
                print(f"Modül {module_name} devre dışı!")
                return False
            
            # Modülü dinamik olarak yükle
            module_path = f"modules.{module_name}"
            module = importlib.import_module(module_path)
            
            # Ana sınıfı bul (genellikle __all__ listesinde)
            if hasattr(module, '__all__') and module.__all__:
                main_class_name = module.__all__[0]
                main_class = getattr(module, main_class_name)
                
                # Modül örneği oluştur
                module_config = config.get_module_config(module_name)
                instance = main_class()
                
                self.loaded_modules[module_name] = {
                    'module': module,
                    'class': main_class,
                    'instance': instance,
                    'config': module_config
                }
                
                print(f"✅ Modül {module_name} başarıyla yüklendi!")
                return True
            else:
                print(f"❌ Modül {module_name} için ana sınıf bulunamadı!")
                return False
                
        except Exception as e:
            print(f"❌ Modül {module_name} yüklenirken hata: {e}")
            return False
    
    def load_all_modules(self):
        """Tüm aktif modülleri yükle"""
        all_modules = config.get_all_modules()
        loaded_count = 0
        
        for module_name in all_modules:
            if self.load_module(module_name):
                loaded_count += 1
        
        print(f"📊 Toplam {loaded_count} modül yüklendi.")
        return loaded_count
    
    def get_module_instance(self, module_name: str):
        """Modül örneğini getir"""
        return self.loaded_modules.get(module_name, {}).get('instance')
    
    def get_module_config(self, module_name: str):
        """Modül yapılandırmasını getir"""
        return self.loaded_modules.get(module_name, {}).get('config')
    
    def list_loaded_modules(self) -> List[str]:
        """Yüklenen modüllerin listesini getir"""
        return list(self.loaded_modules.keys())
    
    def get_module_info(self, module_name: str) -> Dict[str, Any]:
        """Modül bilgilerini getir"""
        module_data = self.loaded_modules.get(module_name, {})
        if module_data:
            return {
                'name': module_name,
                'config': module_data.get('config', {}),
                'loaded': True
            }
        return {'name': module_name, 'loaded': False}
    
    def execute_module_method(self, module_name: str, method_name: str, *args, **kwargs):
        """Modül metodunu çalıştır"""
        instance = self.get_module_instance(module_name)
        if instance and hasattr(instance, method_name):
            method = getattr(instance, method_name)
            return method(*args, **kwargs)
        else:
            raise AttributeError(f"Modül {module_name} için {method_name} metodu bulunamadı!")

# Global modül yöneticisi örneği
module_manager = ModuleManager()
