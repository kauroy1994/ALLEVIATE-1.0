from VirtualAgent import VirtualAgent

if __name__ == '__main__':

    agent = VirtualAgent(modes=['Depression'])
    agent.load_intents() 
    agent.load_patient_data()
    #agent.perform_patient_assessments()
    #agent.build_knowledge_graphs()
    agent.create_vector_db()