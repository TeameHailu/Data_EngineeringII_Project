import pulsar

def convert(word):
    return word.upper()

client = pulsar.Client('pulsar://localhost:6650')
consumer = client.subscribe('conversion-topic', subscription_name='conversion-sub')
producer = client.create_producer('result-topic')

while True:
    msg = consumer.receive()
    try:
        converted = convert(msg.data().decode('utf-8'))
        producer.send(converted.encode('utf-8'))
        consumer.acknowledge(msg)
    except:
        consumer.negative_acknowledge(msg)

client.close()
