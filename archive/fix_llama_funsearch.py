#!/bin/bash
# Script to fix the syntax error in llama_funsearch.py

# Create a backup of the file
cp llama_funsearch.py llama_funsearch.py.broken_backup

# Find line number with the syntax error
error_line=$(grep -n "raisedef" llama_funsearch.py | cut -d ":" -f 1)

if [ -z "$error_line" ]; then
  echo "Error not found, trying alternative method"
  # Try to fix based on known error locations
  sed -i 's/raisedef/raise\n\ndef/g' llama_funsearch.py
else
  echo "Found error on line $error_line"
  # Fix the specific error with proper indentation
  sed -i "${error_line}s/raisedef/raise\n\n    def/g" llama_funsearch.py
fi

# Check for other potential issues
sed -i 's/raisefrom/raise\n\n    from/g' llama_funsearch.py
sed -i 's/raiseimport/raise\n\nimport/g' llama_funsearch.py
sed -i 's/raiseclass/raise\n\nclass/g' llama_funsearch.py

# Verify fix worked
if python -c "import ast; ast.parse(open('llama_funsearch.py').read()); print('Syntax is valid')" 2>/dev/null; then
  echo "Syntax error fixed successfully!"
else
  echo "Syntax error still present. Manual editing may be required."
  # Try a more aggressive fix
  awk '
  {
    # Replace "raise" followed immediately by keywords with proper separation
    gsub(/raise(def|class|import|from|if|for|while|try|except)/, "raise\n\n    \\1");
    print;
  }
  ' llama_funsearch.py > llama_funsearch.py.fixed
  
  mv llama_funsearch.py.fixed llama_funsearch.py
  
  # Check if fix worked
  if python -c "import ast; ast.parse(open('llama_funsearch.py').read()); print('Syntax is valid')" 2>/dev/null; then
    echo "Syntax error fixed with aggressive method!"
  else
    echo "Could not automatically fix syntax error."
    echo "Restoring backup..."
    cp llama_funsearch.py.broken_backup llama_funsearch.py
    
    echo "Please try editing the file manually or using our alternative solution."
  fi
fi 