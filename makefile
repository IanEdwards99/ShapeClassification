install: venv
	. venv/bin/activate; pip3 install -Ur requirements.txt;

venv :
	test -d venv || python3 -m venv venv

clean:
	rm -rf venv
	find -iname "*.pyc" -delete

XOR:
	. venv/bin/activate; python3 XORGate.py
	
Classifier:
	. venv/bin/activate; python3 Classifier.py
	
	
