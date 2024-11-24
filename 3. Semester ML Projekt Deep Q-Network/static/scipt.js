async function generateTask() {
    const difficulty = document.getElementById("difficulty").value;

    try {
        const response = await fetch(`/generate_task?difficulty=${difficulty}`);
        const data = await response.json();

        if (response.ok) {
            document.getElementById("task-output").innerHTML = `<p>Task: ${data.equation}</p><p>Solution: ${data.solution}</p>`;
            if (data.knowledge.concept) {
                document.getElementById("knowledge-output").innerHTML = `
                    <h3>Theoretical Knowledge</h3>
                    <p>${data.knowledge.concept}</p>
                    <ul>
                        ${data.knowledge.steps.map(step => `<li>${step}</li>`).join('')}
                    </ul>
                    <p>Example: ${data.knowledge.example}</p>
                `;
            } else {
                document.getElementById("knowledge-output").innerHTML = `<p>No knowledge available for this task.</p>`;
            }
        } else {
            document.getElementById("task-output").innerHTML = `<p>Error: ${data.error}</p>`;
        }
    } catch (error) {
        document.getElementById("task-output").innerHTML = `<p>Failed to fetch task: ${error.message}</p>`;
    }
}
