const form = document.getElementById("formulario");
const resultado = document.getElementById("resultado");

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const data = Object.fromEntries(new FormData(form).entries());

    resultado.textContent = "Analizando datos...";

    try {
        const res = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(data),
        });

        const result = await res.json();

        if (result.error) {
            resultado.textContent = "❌ Error: " + result.error;
        } else {
            resultado.innerHTML = `<strong>Resultado:</strong> ${result.label}`;
        }
    } catch (err) {
        resultado.textContent = "⚠️ No se pudo conectar con el servidor.";
        console.error(err);
    }
});
