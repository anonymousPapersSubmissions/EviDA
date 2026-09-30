#!/bin/bash

# Create project structure
mkdir -p {config,src/{data,models,losses,training,utils},scripts,data,checkpoints,logs,evaluation_results}


# Create __init__.py files
touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/losses/__init__.py
touch src/training/__init__.py
touch src/utils/__init__.py
touch config/__init__.py

echo "Project structure created successfully!"
echo "Now copy the file contents from the conversation into each file."
echo ""
echo "File list:"
echo "1. requirements.txt"
echo "2. setup.py"
echo "3. README.md"
echo "4. config/config.py"
echo "5. src/utils/logger.py"
echo "6. src/utils/metrics.py"
echo "7. src/data/dataset.py"
echo "8. src/data/data_balancing.py"
echo "9. src/data/samplers.py"
echo "10. src/models/encoders.py"
echo "11. src/models/fusion.py"
echo "12. src/models/classifier.py"
echo "13. src/models/model.py"
echo "14. src/losses/losses.py"
echo "15. src/training/trainer.py"
echo "16. src/training/evaluator.py"
echo "17. scripts/prepare_data.py"
echo "18. scripts/train.py"
echo "19. scripts/evaluate.py"
echo "20. scripts/inference.py"