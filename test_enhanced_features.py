#!/usr/bin/env python3
"""Test script for enhanced ResNet features."""

import torch
import warnings
from models.resnet import resnet50, iresnet100, validate_resnet_model, create_resnet

def main():
    print('🧪 Testing Enhanced ResNet Implementation...')
    
    # Test 1: Device parameter
    try:
        model = resnet50(device='cpu')
        print(f'✅ ResNet-50 created with device: {model.device}')
    except Exception as e:
        print(f'❌ Device parameter test failed: {e}')
    
    # Test 2: Input validation
    try:
        model = resnet50(device='cpu')
        invalid_input = torch.randn(1, 3, 224, 224)  # Wrong size for face recognition
        model(invalid_input)
        print('❌ Input validation failed - should have raised error')
    except ValueError as e:
        print(f'✅ Input validation working: {str(e)[:50]}...')
    except Exception as e:
        print(f'⚠️ Unexpected error in input validation: {e}')
    
    # Test 3: iresnet100 with warning
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            imodel = iresnet100()
            if w and 'iresnet100 is not implemented' in str(w[0].message):
                print('✅ iresnet100 warning displayed correctly')
            else:
                print('⚠️ iresnet100 warning not captured as expected')
    except Exception as e:
        print(f'❌ iresnet100 test failed: {e}')
    
    # Test 4: create_resnet with device
    try:
        model = create_resnet('resnet50', device='cpu')
        print(f'✅ create_resnet with device parameter working')
    except Exception as e:
        print(f'❌ create_resnet device test failed: {e}')
    
    # Test 5: Enhanced validation with gradient check
    try:
        model = resnet50(device='cpu')
        results = validate_resnet_model(model)
        if 'gradient_check' in results and results['gradient_check']:
            print('✅ Gradient flow validation working')
        else:
            print('⚠️ Gradient check not working as expected')
    except Exception as e:
        print(f'❌ Enhanced validation test failed: {e}')
    
    # Test 6: Proper 112x112 input
    try:
        model = resnet50(device='cpu')
        model.eval()
        valid_input = torch.randn(1, 3, 112, 112)
        with torch.no_grad():
            output = model(valid_input)
        print(f'✅ Valid 112x112 input processed successfully: {output.shape}')
    except Exception as e:
        print(f'❌ Valid input test failed: {e}')
    
    print('\n🎉 Enhanced ResNet testing complete!')

if __name__ == "__main__":
    main()
