async function predictBug() {
	const code = document.getElementById("codeInput").value;
	const outputBox = document.getElementById("output");

	outputBox.classList.remove("hidden");
	outputBox.textContent = "⏳ Analyzing...";

	try {
		const res = await fetch("http://127.0.0.1:5000/predict_bug", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ code }),
		});

		const data = await res.json();

		if (data.error) {
			outputBox.textContent = `❌ Error: ${data.error}`;
		} else {
			outputBox.textContent = `✅ Prediction: ${data.prediction.toUpperCase()}\n🎯 Confidence: ${(
				data.confidence * 100
			).toFixed(2)}%`;
		}
	} catch (err) {
		outputBox.textContent = `❌ Connection error: ${err.message}`;
	}
}
