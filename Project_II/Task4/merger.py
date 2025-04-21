import pulsar

client = pulsar.Client('pulsar://localhost:6650')
consumer = client.subscribe('result-topic', subscription_name='result-sub')

result = []
# For demo: assuming we know the number of words
expected_words = 5

for _ in range(expected_words):
    msg = consumer.receive()
    try:
        result.append(msg.data().decode('utf-8'))
        consumer.acknowledge(msg)
    except:
        consumer.negative_acknowledge(msg)

print("Resultant String:", ' '.join(result))
client.close()
