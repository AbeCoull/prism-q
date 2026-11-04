const zoom = document.createElement('dialog');
zoom.className = 'figure-zoom';
zoom.addEventListener('click', () => zoom.close());
document.body.append(zoom);

for (const link of document.querySelectorAll('.content a[href$=".svg"]')) {
    const image = link.querySelector('img');
    if (!image) {
        continue;
    }
    link.addEventListener('click', event => {
        if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey) {
            return;
        }
        event.preventDefault();
        const large = image.cloneNode();
        large.alt = image.alt;
        zoom.replaceChildren(large);
        zoom.showModal();
    });
}
