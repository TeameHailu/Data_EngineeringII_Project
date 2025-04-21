import pulsar

INPUT_STRING = "I want to be capitalized"
client = pulsar.Client('pulsar://localhost:6650')
producer = client.create_producer('conversion-topic')

for word in INPUT_STRING.split():
    producer.send(word.encode('utf-8'))

client.close()
