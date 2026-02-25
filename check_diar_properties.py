import azure.cognitiveservices.speech as speechsdk
print("\n".join([prop for prop in dir(speechsdk.PropertyId) if "Diar" in prop]))
