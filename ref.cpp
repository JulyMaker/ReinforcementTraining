#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

const int SIZE = 5; // Tamaño del laberinto (5x5 en este ejemplo)

class QLearningAgent {
public:
    QLearningAgent() : alpha(0.1), gamma(0.9) {
        // Inicializar la tabla Q con ceros
        for (int i = 0; i < SIZE; ++i) {
            vector<double> row(SIZE, 0.0);
            Q.push_back(row);
        }
        srand(static_cast<unsigned>(time(nullptr)));
    }

    int takeAction(int state) {
        // Elige una acción (movimiento) basada en la exploración o explotación
        if (rand() / static_cast<double>(RAND_MAX) < epsilon) {
            return rand() % 4; // Exploración aleatoria
        } else {
            // Explotación: elige la acción con el valor Q máximo
            int maxAction = 0;
            for (int i = 1; i < 4; ++i) {
                if (Q[state][i] > Q[state][maxAction]) {
                    maxAction = i;
                }
            }
            return maxAction;
        }
    }

    void updateQ(int state, int action, double reward, int nextState) {
        // Actualizar la tabla Q usando la regla de actualización Q-learning
        Q[state][action] += alpha * (reward + gamma * getMaxQ(nextState) - Q[state][action]);
    }

private:
    double alpha; // Tasa de aprendizaje
    double gamma; // Factor de descuento
    double epsilon = 0.1; // Probabilidad de exploración

    vector<vector<double>> Q; // Tabla Q

    double getMaxQ(int nextState) {
        // Obtiene el valor Q máximo para el siguiente estado
        return *max_element(Q[nextState].begin(), Q[nextState].end());
    }
};

class Environment {
public:
    Environment() : agentPosition(0) {
        // Inicializa el entorno con recompensas positivas y negativas
        rewards = {
            {0, 1, -1, 0, 0},
            {0, -1, 0, 0, 0},
            {0, 0, 1, -1, 0},
            {0, 0, 0, -1, 1},
            {0, 0, 0, 0, 0}
        };
    }

    int getAgentPosition() const {
        return agentPosition;
    }

    double takeAction(int action) {
        // El agente toma una acción (mueve) y recibe una recompensa
        int nextState;
        switch (action) {
            case 0: // Arriba
                nextState = max(agentPosition - SIZE, 0);
                break;
            case 1: // Abajo
                nextState = min(agentPosition + SIZE, SIZE * SIZE - 1);
                break;
            case 2: // Izquierda
                nextState = max(agentPosition - 1, 0);
                break;
            case 3: // Derecha
                nextState = min(agentPosition + 1, SIZE * SIZE - 1);
                break;
            default:
                cerr << "Acción no válida" << endl;
                exit(EXIT_FAILURE);
        }

        double reward = rewards[nextState / SIZE][nextState % SIZE];
        agentPosition = nextState;
        return reward;
    }

private:
    int agentPosition;
    vector<vector<double>> rewards;
};

int main() {
    QLearningAgent agent;
    Environment environment;

    // Entrenamiento del agente
    for (int episode = 0; episode < 1000; ++episode) {
        int state = environment.getAgentPosition();
        while (state != SIZE * SIZE - 1) { // Continuar hasta llegar al estado objetivo
            int action = agent.takeAction(state);
            double reward = environment.takeAction(action);
            int nextState = environment.getAgentPosition();
            agent.updateQ(state, action, reward, nextState);
            state = nextState;
        }
    }

    // Prueba del agente entrenado
    int state = environment.getAgentPosition();
    cout << "Estado inicial: " << state << endl;

    while (state != SIZE * SIZE - 1) {
        int action = agent.takeAction(state);
        double reward = environment.takeAction(action);
        int nextState = environment.getAgentPosition();

        cout << "Acción tomada: " << action << ", Nuevo estado: " << nextState << ", Recompensa: " << reward << endl;

        state = nextState;
    }

    return 0;
}
