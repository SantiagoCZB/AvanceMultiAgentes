from fastapi import FastAPI, Request, Query
from fastapi import Request
#import pydub as pd
#import simpleaudio as sa
#from playsound import playsound


app = FastAPI()

tractor_path = []

#from pydub import AudioSegment
#import simpleaudio as sa

app = FastAPI()

tractor_path = []

# Load the audio file
#audio = AudioSegment.from_wav('soundtrack.wav')
#play_obj = None

#def play_audio():
#    global play_obj
#    if play_obj and play_obj.is_playing():
#        play_obj.stop()
#    play_obj = sa.play_buffer(audio.raw_data, num_channels=audio.channels, bytes_per_sample=audio.sample_width, sample_rate=audio.frame_rate)

#def stop_audio():
#    global play_obj
#    if play_obj and play_obj.is_playing():
#        play_obj.stop()


def create_tractor_lists(n_tractors):
    for i in range(n_tractors):
        tractor_path.append([])
    return tractor_path

@app.post("/send_coordinates")
async def receive_coordinates(request: Request, id : int = Query(...)):
    coordinates = await request.json()
    x = coordinates['x']
    y = coordinates['y']

    tractor_path[id].append((x, y))

    print(coordinates)

    return {"message": "Coordinates received", "x": x, "y": y}

@app.get("/get_coordinates")
async def get_coordinates(id: int = Query(...)):
    return tractor_path[id]

@app.get("/get_tractor_path")
async def get_tractor_path():
    return {"tractor_path": tractor_path}

@app.get("/check_connection")
async def check_connection_endpoint(n_tractors: int = Query(...)):
    create_tractor_lists(n_tractors)
    #play_audio()
    return {"message": "Connection OK"}

#@app.get("/stop_connection")
#async def stop_connection_endpoint():
#    stop_audio()
#    return {"message": "Connection stopped"}

if __name__ == "__main__":
#    playsound('./sfx.wav')
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# http://localhost:8000/check_connection?n_tractors=4