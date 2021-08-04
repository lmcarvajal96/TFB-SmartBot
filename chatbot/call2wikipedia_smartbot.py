import json
import wikipedia as wiki
def lambda_handler(event, context):
 #keyboard = input('Hi, What do you want me to look up in Wikipedia?')
 text = event.get("inputTranscript")
 text = wiki.summary(text)
 return {
 "messages": [
  {
   "content": text,
   "contentType": "PlainText"
  }
 ],
 "sessionState": {
  "dialogAction": {
   "type": "Close",
  },
  "intent": {
   "name": "wikiSearch",
   "state": "InProgress",
   "confirmationState": "None"
  },
  "originatingRequestId": "432cd4ad-0dad-4cd8-a8fb-c64b89d711c1"
 },
 "interpretations": [
  {
   "nluConfidence": {
    "score": 1
   },
   "intent": {
    "name": "wikiSearch",
    "state": "InProgress",
    "confirmationState": "None"
   }
  },
  {
   "intent": {
    "name": "FallbackIntent",
    "slots": {}
   }
  }
 ],
 "sessionId": "271840027647435"
}